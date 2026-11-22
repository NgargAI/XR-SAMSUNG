// Path Finder - IMU-only trajectory estimation under sensor degradation.
//
//   pathfinder <imu0/data.csv> <estimated_trajectory.txt> [options]
//
// Input : EuRoC IMU csv  (t[ns], wx, wy, wz [rad/s], ax, ay, az [m/s^2])
// Output: TUM trajectory (t[s] tx ty tz qx qy qz qw), exactly one pose per input row.
//
// Design (fits the 8 MB / 20 ms-per-second limits):
//   * Pure streaming: one line in, one line out, O(1) memory (~1 MB RSS static build).
//   * Strapdown INS: quaternion attitude + velocity + position, midpoint integration
//     with the REAL timestamp deltas (so dropped frames are integrated correctly).
//   * Attitude: Mahony-style gravity correction, gated by |a|~g and 
//   * Zero-velocity updates (only when the IMU looks still AND the speed estimate
//     is already small) that also re-estimate the gyro bias.
//   * Failure detection (NaN / zeros / absurd values / frozen identical readings):
//     the filter stops trusting the IMU and coasts (constant velocity, held
//     attitude) from the last good state -> the "blind 5 s" dead-reckoning phase.
//
// Frame: world z = up, yaw = 0 and position = 0 at the first sample.  Output is the
// IMU body pose in that world frame.  (Scoring with evo should use alignment.)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct V3 { double x, y, z; };
struct Q  { double w, x, y, z; };

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline Q qmul(const Q& a, const Q& b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
inline Q qnormalize(Q q) {
  double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-12) return {1, 0, 0, 0};
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}
inline Q qexp(V3 r) {  // rotation vector -> quaternion
  double a = norm(r);
  if (a < 1e-10) return qnormalize({1.0, r.x * 0.5, r.y * 0.5, r.z * 0.5});
  double s = std::sin(0.5 * a) / a;
  return {std::cos(0.5 * a), r.x * s, r.y * s, r.z * s};
}
inline V3 qrot(const Q& q, V3 v) {  // body -> world
  V3 qv{q.x, q.y, q.z};
  V3 t = cross(qv, v) * 2.0;
  return v + t * q.w + cross(qv, t);
}
inline V3 qrot_inv(const Q& q, V3 v) { return qrot({q.w, -q.x, -q.y, -q.z}, v); }

// ----------------------------------------------------------------------------
struct Cfg {
  double kp = 4.0;            // attitude gravity-correction gain  [1/s]
  double ki = 1.0;            // gyro-bias gain from gravity error
  double acc_tol = 1.0;       // |a|-g tolerance for gravity correction [m/s^2]
  bool zupt = true;
  double zupt_speed = 3.0;    // only zero velocity if already slower than this [m/s]
  double zupt_time = 0.15;    // IMU must look still for this long [s]
  double dead_decay = 0.0;    // velocity decay while coasting [1/s] (0 = const velocity)
  double smooth = 2.0;        // offline drift removal: zero-phase high-pass time constant [s] (0 = off)
  double vleak = 0.2;         // velocity leak while the IMU is healthy [1/s] (bounded-motion prior; 0 = off)
  double vmax = 40.0;         // sanity clamp [m/s]
  int stuck_n = 20;           // identical consecutive readings => sensor frozen
  int bad_n = 20;             // consecutive invalid readings => sensor dead
  double init_time = 0.6;     // stationary start-up window [s]
};

struct Imu {
  int64_t t;  // ns
  V3 w, a;
  bool ok;    // parsed with all 6 finite numbers
};

constexpr int kMaxInit = 400;
constexpr int kMaxPending = 256;
constexpr double kG0 = 9.80665;

class Ins {
 public:
  explicit Ins(const Cfg& c, FILE* out) : c_(c), out_(out) {}

  void push(const Imu& s) {
    if (!inited_) {
      init_buf_[n_init_++] = s;
      if (n_init_ >= kMaxInit || (double)(s.t - init_buf_[0].t) * 1e-9 >= c_.init_time) finish_init();
      return;
    }
    feed(s);
  }

  void finish() {
    if (!inited_) finish_init();
    flush_pending(false);
  }

 private:
  // ---- configuration / IO ----
  Cfg c_;
  FILE* out_;

  // ---- init ----
  bool inited_ = false;
  Imu init_buf_[kMaxInit];
  int n_init_ = 0;

  // ---- state ----
  Q q_{1, 0, 0, 0};
  V3 p_{0, 0, 0}, v_{0, 0, 0};
  V3 bg_{0, 0, 0};   // gyro bias
  V3 ei_{0, 0, 0};   // integral correction from gravity error
  double g_ = kG0;
  double dt0_ = 0.005;
  int64_t tlast_ = 0;
  bool have_t_ = false;

  // last good measurement
  V3 mw_{0, 0, 0}, ma_{0, 0, kG0};
  bool have_meas_ = false;

  // failure handling
  bool dead_ = false;
  int bad_run_ = 0;
  int64_t pend_t_[kMaxPending];
  int n_pend_ = 0;

  // stats for adaptive logic
  double ema_g2_ = 0, ema_a2_ = 0;   // gyro energy / accel variance about slow mean
  V3 gm_{0, 0, 0};                   // slow mean of raw gyro
  V3 am_{0, 0, kG0};                 // slow mean of raw accel
  double thr_g_ = 0.01, thr_a_ = 0.05;
  double still_t_ = 0;

  // ---------------------------------------------------------------- output
  void emit(int64_t t) {
    long long sec = t / 1000000000LL, ns = t % 1000000000LL;
    if (ns < 0) { ns += 1000000000LL; sec -= 1; }
    std::fprintf(out_, "%lld.%09lld %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", sec, ns, p_.x, p_.y, p_.z,
                 q_.x, q_.y, q_.z, q_.w);
  }

  static bool plausible(const Imu& s) {
    if (!s.ok) return false;
    double an = norm(s.a), wn = norm(s.w);
    return an > 2.0 && an < 160.0 && wn < 35.0;
  }

  // ---------------------------------------------------------------- init
  void finish_init() {
    inited_ = true;
    // Median sample period.
    double dts[kMaxInit];
    int nd = 0;
    for (int i = 1; i < n_init_; ++i) {
      double d = (double)(init_buf_[i].t - init_buf_[i - 1].t) * 1e-9;
      if (d > 0 && d < 0.2) dts[nd++] = d;
    }
    if (nd > 0) {
      std::nth_element(dts, dts + nd / 2, dts + nd);
      dt0_ = dts[nd / 2];
    }
    // Stationary statistics over good samples.
    V3 mw{0, 0, 0}, ma{0, 0, 0};
    int ng = 0;
    for (int i = 0; i < n_init_; ++i)
      if (plausible(init_buf_[i])) { mw = mw + init_buf_[i].w; ma = ma + init_buf_[i].a; ++ng; }
    if (ng > 0) {
      mw = mw * (1.0 / ng);
      ma = ma * (1.0 / ng);
      double sg = 0, sa = 0;
      for (int i = 0; i < n_init_; ++i) {
        if (!plausible(init_buf_[i])) continue;
        V3 dw = init_buf_[i].w - mw;
        sg += dot(dw, dw);
        V3 da = init_buf_[i].a - ma;
        sa += dot(da, da);
      }
      sg = std::sqrt(sg / ng);
      sa = std::sqrt(sa / ng);
      bool still = sg < 0.08;  // otherwise do not trust the mean gyro as a bias
      if (still) bg_ = mw;
      double gn = norm(ma);
      g_ = (gn > 9.0 && gn < 10.6) ? gn : kG0;
      // initial attitude: rotate measured "up" (body) onto world +z
      V3 up = ma * (1.0 / std::max(gn, 1e-9));
      V3 ez{0, 0, 1};
      V3 ax = cross(up, ez);
      double s = norm(ax), cth = clampd(dot(up, ez), -1.0, 1.0);
      if (s < 1e-9) q_ = (cth > 0) ? Q{1, 0, 0, 0} : Q{0, 1, 0, 0};
      else q_ = qexp(ax * (std::atan2(s, cth) / s));
      thr_g_ = std::max(0.01, 4.0 * sg);
      thr_a_ = std::max(0.05, 4.0 * sa);
      gm_ = bg_;
      am_ = ma;
    }
    for (int i = 0; i < n_init_; ++i) feed(init_buf_[i]);
  }

  // ---------------------------------------------------------------- one INS step
  void propagate(double h, V3 w, V3 a) {
    double an = norm(a);
    // gravity-direction attitude correction (Mahony)
    V3 up_b = qrot_inv(q_, V3{0, 0, 1});
    V3 e = cross(a * (1.0 / std::max(an, 1e-9)), up_b);
    double dev = (an - g_) / c_.acc_tol;
    double gate = 1.0 / (1.0 + dev * dev * dev * dev);
    double kg = gate;
    ei_ = ei_ + e * (c_.ki * kg * h);
    ei_.x = clampd(ei_.x, -0.05, 0.05);
    ei_.y = clampd(ei_.y, -0.05, 0.05);
    ei_.z = clampd(ei_.z, -0.05, 0.05);
    V3 wc = w - bg_ + ei_ + e * (c_.kp * kg);

    Q qmid = qnormalize(qmul(q_, qexp(wc * (0.5 * h))));
    q_ = qnormalize(qmul(q_, qexp(wc * h)));
    V3 aw = qrot(qmid, a) - V3{0, 0, g_};
    V3 vn = v_ + aw * h;
    if (c_.vleak > 0) vn = vn * std::exp(-c_.vleak * h);
    double sp = norm(vn);
    if (sp > c_.vmax) vn = vn * (c_.vmax / sp);
    p_ = p_ + (v_ + vn) * (0.5 * h);
    v_ = vn;

    // ---- stationarity / noise bookkeeping
    double al = 1.0 - std::exp(-h / 0.2);
    V3 wr = w - bg_;
    ema_g2_ += al * (dot(wr, wr) - ema_g2_);
    V3 da = a - am_;
    ema_a2_ += al * (dot(da, da) - ema_a2_);
    am_ = am_ + (a - am_) * (1.0 - std::exp(-h / 1.0));
    gm_ = gm_ + (w - gm_) * al;
    bool still = std::sqrt(ema_g2_) < thr_g_ && std::sqrt(ema_a2_) < thr_a_;
    still_t_ = still ? still_t_ + h : 0.0;
    if (c_.zupt && still_t_ >= c_.zupt_time && norm(v_) < c_.zupt_speed) {
      v_ = {0, 0, 0};
      bg_ = bg_ + (gm_ - bg_) * std::min(1.0, h / 2.0);
    }
  }

  // Integrate from tlast_ to cur.t using measurement linearly interpolated between the
  // previous good sample and this one (handles dropped frames).
  void integrate_to(const Imu& cur) {
    double dt = (double)(cur.t - tlast_) * 1e-9;
    if (dt <= 0) { return; }
    V3 w0 = have_meas_ ? mw_ : cur.w, a0 = have_meas_ ? ma_ : cur.a;
    int n = (int)clampd(std::floor(dt / dt0_ + 0.5), 1.0, 1000.0);
    double h = dt / n;
    for (int k = 0; k < n; ++k) {
      double f0 = (double)k / n, f1 = (double)(k + 1) / n;
      double fm = 0.5 * (f0 + f1);  // midpoint measurement
      V3 w = w0 + (cur.w - w0) * fm, a = a0 + (cur.a - a0) * fm;
      propagate(h, w, a);
      if (k + 1 < n) { ma_ = a; }
    }
    mw_ = cur.w;
    ma_ = cur.a;
    have_meas_ = true;
    tlast_ = cur.t;
  }

  void coast_to(int64_t t) {
    double dt = (double)(t - tlast_) * 1e-9;
    if (dt > 0) {
      p_ = p_ + v_ * dt;
      if (c_.dead_decay > 0) v_ = v_ * std::exp(-c_.dead_decay * dt);
      tlast_ = t;
    }
    still_t_ = 0;
  }

  // ---------------------------------------------------------------- per-sample logic
  void feed(const Imu& s) {
    if (!have_t_) {
      have_t_ = true;
      tlast_ = s.t;
      if (plausible(s)) { mw_ = s.w; ma_ = s.a; have_meas_ = true; }
      emit(s.t);
      return;
    }
    if (dead_) { coast_to(s.t); emit(s.t); return; }

    if (!plausible(s)) {
      // Invalid reading: flush any frozen run first, then coast on the last good state.
      flush_pending(false);
      coast_to(s.t);
      have_meas_ = false;
      if (++bad_run_ >= c_.bad_n) dead_ = true;
      emit(s.t);
      return;
    }
    bad_run_ = 0;

    // Frozen sensor?  (bit-identical to the previous reading)
    if (have_meas_ && s.w.x == mw_.x && s.w.y == mw_.y && s.w.z == mw_.z && s.a.x == ma_.x &&
        s.a.y == ma_.y && s.a.z == ma_.z) {
      pend_t_[n_pend_++] = s.t;
      if (n_pend_ >= c_.stuck_n || n_pend_ >= kMaxPending) {
        // Declared dead: state was NOT advanced by the frozen samples, so just coast.
        dead_ = true;
        flush_pending(true);
      }
      return;
    }
    flush_pending(false);
    step(s);
  }

  void step(const Imu& s) {
    if (s.t <= tlast_) { emit(s.t); return; }  // duplicate / out-of-order stamp
    integrate_to(s);
    emit(s.t);
  }

  // Frozen run ended without being declared a failure (or EOF): integrate normally.
  // If it WAS a failure, coast through the held timestamps.
  void flush_pending(bool as_dead) {
    if (n_pend_ == 0) return;
    int n = n_pend_;
    n_pend_ = 0;
    for (int i = 0; i < n; ++i) {
      if (as_dead) { coast_to(pend_t_[i]); emit(pend_t_[i]); }
      else {
        Imu h{pend_t_[i], mw_, ma_, true};
        step(h);
      }
    }
  }
};

// ----------------------------------------------------------------------------
bool parse_line(char* s, Imu& o) {
  while (*s == ' ' || *s == '\t') ++s;
  if (*s == '#' || *s == '\0' || *s == '\n' || *s == '\r') return false;
  char* e;
  long long t = std::strtoll(s, &e, 10);
  if (e == s) return false;  // header / garbage
  if (*e == '.' || *e == 'e' || *e == 'E') { t = std::llround(std::strtod(s, &e)); }
  o.t = t;
  double v[6];
  bool ok = true;
  char* p = e;
  for (int i = 0; i < 6; ++i) {
    if (*p != ',') { ok = false; v[i] = 0; continue; }
    ++p;
    v[i] = std::strtod(p, &e);
    if (e == p) {  // empty / non-numeric field
      ok = false;
      v[i] = 0;
      char* c = std::strchr(p, ',');
      p = c ? c : p + std::strlen(p);
    } else {
      p = e;
    }
    if (!std::isfinite(v[i])) { ok = false; v[i] = 0; }
  }
  o.w = {v[0], v[1], v[2]};
  o.a = {v[3], v[4], v[5]};
  o.ok = ok;
  return true;
}


// Pass 2 (offline drift removal). The IMU path is accurate over seconds but drifts slowly.
// A zero-phase (forward+backward) first-order low-pass with time constant tau isolates the
// drift, which is subtracted. Forward low-pass runs at full rate; it is stored every
// kDecim seconds (a few hundred kB even for hours of data), the backward pass runs on
// that, and the result is interpolated while the final file is written.
// Assumption: bounded motion (a room). Disable with --no-smooth for long trips.
struct Knot { double t, x, y, z; };
constexpr double kDecim = 0.1;

// Reads "t x y z rest"; keeps the timestamp text verbatim (no precision loss).
bool read_pose(FILE* f, double& t, double* p, char* ts, char* rest, size_t rn) {
  char l[256];
  if (!std::fgets(l, sizeof l, f)) return false;
  int n1 = 0, off = 0;
  if (std::sscanf(l, "%39s%n", ts, &n1) < 1) return false;
  t = std::atof(ts);
  if (std::sscanf(l + n1, "%lf %lf %lf %n", &p[0], &p[1], &p[2], &off) < 3) return false;
  std::snprintf(rest, rn, "%s", l + n1 + off);
  return true;
}

void smooth_pass(FILE* tmp, FILE* out, double tau) {
  std::rewind(tmp);
  std::vector<Knot> k;
  double t, p[3], lp[3] = {0, 0, 0}, tl = 0;
  char rest[200], ts[40];
  bool first = true;
  while (read_pose(tmp, t, p, ts, rest, sizeof rest)) {
    if (first) { lp[0] = p[0]; lp[1] = p[1]; lp[2] = p[2]; first = false; }
    else {
      double a = 1.0 - std::exp(-std::max(t - tl, 0.0) / tau);
      for (int j = 0; j < 3; ++j) lp[j] += a * (p[j] - lp[j]);
    }
    tl = t;
    if (k.empty() || t - k.back().t >= kDecim) k.push_back({t, lp[0], lp[1], lp[2]});
  }
  for (size_t i = k.size(); i-- > 1;) {  // backward pass
    double a = 1.0 - std::exp(-std::max(k[i].t - k[i - 1].t, 0.0) / tau);
    k[i - 1].x = k[i].x + a * (k[i - 1].x - k[i].x);
    k[i - 1].y = k[i].y + a * (k[i - 1].y - k[i].y);
    k[i - 1].z = k[i].z + a * (k[i - 1].z - k[i].z);
  }
  std::rewind(tmp);
  size_t j = 0;
  while (read_pose(tmp, t, p, ts, rest, sizeof rest)) {
    while (j + 1 < k.size() && k[j + 1].t < t) ++j;
    double lx = k[j].x, ly = k[j].y, lz = k[j].z;
    if (j + 1 < k.size() && t > k[j].t) {
      double f = std::min(1.0, (t - k[j].t) / std::max(k[j + 1].t - k[j].t, 1e-12));
      lx += f * (k[j + 1].x - lx); ly += f * (k[j + 1].y - ly); lz += f * (k[j + 1].z - lz);
    }
    std::fprintf(out, "%s %.6f %.6f %.6f %s", ts, p[0] - lx, p[1] - ly, p[2] - lz, rest);
  }
  std::fclose(tmp);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s imu0/data.csv estimated_trajectory.txt [--kp X] [--ki X] [--no-zupt]\n"
                 "          [--zupt-speed X] [--zupt-time X] [--smooth SEC | --no-smooth] [--vleak X] [--dead-decay X] [--stuck-n N] [--bad-n N] [--report]\n", argv[0]);
    return 2;
  }
  Cfg cfg;
  bool report = false;
  for (int i = 3; i < argc; ++i) {
    auto arg = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (arg("--kp")) cfg.kp = std::atof(argv[++i]);
    else if (arg("--ki")) cfg.ki = std::atof(argv[++i]);
    else if (arg("--zupt-speed")) cfg.zupt_speed = std::atof(argv[++i]);
    else if (arg("--zupt-time")) cfg.zupt_time = std::atof(argv[++i]);
    else if (arg("--smooth")) cfg.smooth = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-smooth") == 0) cfg.smooth = 0;
    else if (arg("--vleak")) cfg.vleak = std::atof(argv[++i]);
    else if (arg("--dead-decay")) cfg.dead_decay = std::atof(argv[++i]);
    else if (arg("--stuck-n")) cfg.stuck_n = std::atoi(argv[++i]);
    else if (arg("--bad-n")) cfg.bad_n = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--no-zupt") == 0) cfg.zupt = false;
    else if (std::strcmp(argv[i], "--report") == 0) report = true;
    else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  FILE* in = std::strcmp(argv[1], "-") == 0 ? stdin : std::fopen(argv[1], "r");
  FILE* out = std::strcmp(argv[2], "-") == 0 ? stdout : std::fopen(argv[2], "w");
  if (!in || !out) { std::perror("open"); return 1; }
  static char ibuf[1 << 14], obuf[1 << 15];
  std::setvbuf(in, ibuf, _IOFBF, sizeof ibuf);
  std::setvbuf(out, obuf, _IOFBF, sizeof obuf);

  // Pass 1: causal INS -> temporary file (constant memory).
  FILE* tmp = cfg.smooth > 0 ? std::tmpfile() : out;
  if (!tmp) { std::perror("tmpfile"); return 1; }
  Ins ins(cfg, tmp);
  char line[1024];
  Imu s;
  while (std::fgets(line, sizeof line, in)) {
    if (parse_line(line, s)) ins.push(s);
  }
  ins.finish();
  if (cfg.smooth > 0) smooth_pass(tmp, out, cfg.smooth);
  std::fflush(out);
  if (in != stdin) std::fclose(in);
  if (out != stdout) std::fclose(out);
  if (report) {  // peak resident memory of this process (Linux)
    FILE* f = std::fopen("/proc/self/status", "r");
    char l[256];
    while (f && std::fgets(l, sizeof l, f))
      if (!std::strncmp(l, "VmHWM", 5)) std::fprintf(stderr, "%s", l);
    if (f) std::fclose(f);
  }
  return 0;
}
