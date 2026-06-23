scripts\atmosphere-qlut-gen\generate_atmosphere_luts.ps1 `
      -QLTransExe .\build\src\tools\Release\QLTrans.exe `
      -OutDir .\assets\luts\modtran `
      -MaxJobs 16 `
      -Locations @(
          @{lat=39.9; lon=116.4},
          @{lat=52.9; lon=122.5},
          @{lat=1.3; lon=103.8},
          @{lat=-37.8; lon=145.0},
          @{lat=21.3; lon=-157.8},
          @{lat=35.5; lon=135.8},
          @{lat=69.0; lon=33.1},
          @{lat=8.8; lon=-79.7},
          @{lat=38.9; lon=-76.9},
          @{lat=51.5; lon=-0.1},
          @{lat=-34.5; lon=-58.3},
          @{lat=38.6; lon=-121.3},
          @{lat=-33.9; lon=18.4},
          @{lat=34.2; lon=108.9},
          @{lat=26.6; lon=56.5},
          @{lat=19.1; lon=72.9},
          @{lat=-1.3; lon=36.8},
          @{lat=-15.8; lon=-47.9},
          @{lat=64.1; lon=-21.9},
          @{lat=-17.8; lon=177.9},
          @{lat=52.1; lon=31.8},
          @{lat=38.8; lon=120.0},
          @{lat=35.8; lon=122.2},
          @{lat=29.6; lon=125.7},
          @{lat=23.5; lon=122.3},
          @{lat=19.8; lon=112.8},
          @{lat=15.0; lon=117.7},
          @{lat=7.3; lon=109.2},
          @{lat=43.2; lon=76.9},
          @{lat=-12.3; lon=130.8},
          @{lat=13.5; lon=144.5},
          @{lat=-82.6; lon=-170.6},
          @{lat=61.1; lon=-149.5},
          @{lat=74.0; lon=-100.5},
          @{lat=7.1; lon=171.4},
          @{lat=51.8; lon=-176.6},
          @{lat=-31.9; lon=115.8},
          @{lat=7.1; lon=99.1},
          @{lat=4.5; lon=3.5},
          @{lat=-69.4; lon=76.4}
      ) `
      -Times @(
          @{year=2026; month=3; day=21; hour=8; utc=0},
          @{year=2026; month=3; day=21; hour=14; utc=0},
          @{year=2026; month=3; day=21; hour=20; utc=0},
          @{year=2026; month=3; day=21; hour=2; utc=0},
          @{year=2026; month=6; day=22; hour=8; utc=0},
          @{year=2026; month=6; day=22; hour=14; utc=0},
          @{year=2026; month=6; day=22; hour=20; utc=0},
          @{year=2026; month=6; day=22; hour=2; utc=0},
          @{year=2026; month=9; day=23; hour=8; utc=0},
          @{year=2026; month=9; day=23; hour=14; utc=0},
          @{year=2026; month=9; day=23; hour=20; utc=0},
          @{year=2026; month=9; day=23; hour=2; utc=0},
          @{year=2026; month=12; day=22; hour=8; utc=0},
          @{year=2026; month=12; day=22; hour=14; utc=0},
          @{year=2026; month=12; day=22; hour=20; utc=0},
          @{year=2026; month=12; day=22; hour=2; utc=0}
      ) `
      -Models @(2, 3, 6) `
      -Weathers @(0, 5) `
      -Hazes @(1, 4, 5)