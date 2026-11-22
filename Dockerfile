# Path Finder (Track 03): reproducible build of the C++ estimator.
#   docker build -t pathfinder .
#   docker run --rm -v "$PWD/data_dir:/data" -v "$PWD/out:/out" pathfinder
# /data must contain imu0/data.csv; the result is written to /out/estimated_trajectory.txt
FROM gcc:13 AS build
WORKDIR /src
COPY src/main.cpp .
RUN g++ -O2 -std=c++17 -static -o pathfinder main.cpp

FROM debian:bookworm-slim
RUN useradd --create-home --uid 10001 runner
COPY --from=build /src/pathfinder /usr/local/bin/pathfinder
USER runner
ENTRYPOINT ["pathfinder"]
CMD ["/data/imu0/data.csv", "/out/estimated_trajectory.txt"]
