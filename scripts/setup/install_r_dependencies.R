# One-time package install script
user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", paste(R.version$major, R.version$minor, sep="."))
if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
.libPaths(user_lib)

install.packages("remotes", lib = user_lib, repos = "https://cloud.r-project.org")
install.packages("readr", lib = user_lib, repos = "https://cloud.r-project.org")
remotes::install_github("sportsdataverse/hoopR", lib = user_lib)
