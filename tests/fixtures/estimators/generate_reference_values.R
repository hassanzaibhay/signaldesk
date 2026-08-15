.libPaths(c("/rlib", .libPaths()))
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/bookworm/latest"))
if (!requireNamespace("epitools", quietly = TRUE)) {
  install.packages("epitools", lib = "/rlib", quiet = TRUE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", lib = "/rlib", quiet = TRUE)
}
library(epitools)
library(jsonlite)
library(openEBGM)
set.seed(20260815)

# PhViD's decision rules call LBE, which is no longer installable. The estimator
# arithmetic runs before any decision rule and does not touch it, so LBE is
# stubbed to a constant. Only the statistic columns are read out of these calls.
LBE <- function(p, plot.type = "none") list(pi0 = 1)
for (f in list.files("PhViD/R", full.names = TRUE)) source(f)

# ---------------------------------------------------------------------------
# Part 1: 2x2 tables through epitools (ROR, PRR), stats::chisq.test (Yates),
# and PhViD::BCPNN (IC posterior mean and variance).
# ---------------------------------------------------------------------------

tables <- list(
  list(name = "moderate_signal",      a = 25,   b = 1975,  c = 1000,   d = 97000),
  list(name = "minimum_count",        a = 3,    b = 97,    c = 500,    d = 99400),
  list(name = "strong_signal",        a = 100,  b = 100,   c = 100,    d = 99700),
  list(name = "below_null",           a = 5,    b = 995,   c = 2000,   d = 97000),
  list(name = "large_counts",         a = 5000, b = 50000, c = 100000, d = 9845000),
  list(name = "small_balanced",       a = 12,   b = 18,    c = 30,     d = 40),
  list(name = "rare_drug_rare_event", a = 4,    b = 6,     c = 20,     d = 999970)
)

rows <- list()
for (t in tables) {
  a <- t$a; b <- t$b; c_ <- t$c; d <- t$d
  N <- a + b + c_ + d
  n1. <- a + b
  n.1 <- a + c_

  # epitools wants unexposed/no-outcome in the first row and column.
  m <- matrix(c(d, c_, b, a), nrow = 2, byrow = TRUE)
  or <- epitools::oddsratio.wald(m)$measure[2, ]
  rr <- epitools::riskratio.wald(m)$measure[2, ]

  # Yates-corrected chi-squared on the same table.
  chi <- suppressWarnings(
    stats::chisq.test(matrix(c(a, b, c_, d), nrow = 2, byrow = TRUE), correct = TRUE)
  )

  db <- list(
    data = matrix(c(a, n1., n.1), nrow = 1),
    N = N,
    L = data.frame(drug = "D", event = "E", stringsAsFactors = FALSE)
  )
  bc <- BCPNN(db, RANKSTAT = 2, DECISION = 2, DECISION.THRES = 1)
  ic_lower <- bc$ALLSIGNALS[1, "Q_0.025(log(IC))"]

  # The posterior mean and variance themselves, from the same posterior PhViD
  # uses, recovered by inverting its reported lower bound is not possible, so
  # they are recomputed here from the package's own parameterisation.
  p1 <- 1 + n1.; p2 <- 1 + N - n1.
  q1 <- 1 + n.1; q2 <- 1 + N - n.1
  r1 <- 1 + a;   r2b <- N - a - 1 + (2 + N)^2 / (q1 * p1)
  eic <- log(2)^(-1) * (digamma(r1) - digamma(r1 + r2b) -
                          (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2)))
  vic <- log(2)^(-2) * (trigamma(r1) - trigamma(r1 + r2b) +
                          (trigamma(p1) - trigamma(p1 + p2) + trigamma(q1) - trigamma(q1 + q2)))
  stopifnot(abs(qnorm(0.025, eic, sqrt(vic)) - ic_lower) < 1e-9)

  rows[[length(rows) + 1]] <- list(
    name = t$name, a = a, b = b, c = c_, d = d,
    ror = unname(or["estimate"]),
    ror_lower = unname(or["lower"]),
    ror_upper = unname(or["upper"]),
    prr = unname(rr["estimate"]),
    prr_lower = unname(rr["lower"]),
    prr_upper = unname(rr["upper"]),
    chi2_yates = unname(chi$statistic),
    ic_posterior_mean = eic,
    ic_variance = vic,
    ic_lower_2p5 = ic_lower
  )
}

# ---------------------------------------------------------------------------
# Part 2: MGPS through openEBGM on its shipped CAERS data.
# ---------------------------------------------------------------------------

data(caers)
proc <- processRaw(caers)
squashed <- squashData(proc, bin_size = 300, keep_pts = 10)
squashed <- squashData(squashed, count = 2, bin_size = 13, keep_pts = 10)

theta_init <- data.frame(
  alpha1 = c(0.2, 0.5, 1),
  beta1  = c(0.1, 0.5, 1),
  alpha2 = c(2,   2,   3),
  beta2  = c(4,   2,   3),
  p      = c(1 / 3, 0.1, 0.2)
)
hyper <- autoHyper(data = squashed, theta_init = theta_init)
theta_hat <- hyper$estimates

qn <- Qn(theta_hat, N = proc$N, E = proc$E)
EBGM <- ebgm(theta_hat, N = proc$N, E = proc$E, qn = qn, digits = 6)
EBGM05 <- quantBisect(5, theta_hat, N = proc$N, E = proc$E, qn = qn, digits = 6)

# The full observed table is the reference for the fit; a deterministic sample
# of pairs is the reference for the per-pair posterior arithmetic.
idx <- sort(sample(seq_len(nrow(proc)), 300))

mgps <- list(
  package = "openEBGM",
  package_version = as.character(packageVersion("openEBGM")),
  dataset = "caers",
  n_pairs = nrow(proc),
  n_squashed_rows = nrow(squashed),
  likelihood = "negLLsquash, conditional on N >= 1 (zero counts excluded)",
  theta_hat = list(
    alpha1 = theta_hat[1], beta1 = theta_hat[2],
    alpha2 = theta_hat[3], beta2 = theta_hat[4], p = theta_hat[5]
  ),
  neg_log_likelihood_squashed = negLLsquash(theta_hat, ni = squashed$N, ei = squashed$E,
                                            wi = squashed$weight, N_star = 1),
  neg_log_likelihood_sample = negLL(theta_hat, N = proc$N[idx], E = proc$E[idx], N_star = 1),
  squashed = list(N = squashed$N, E = squashed$E, weight = squashed$weight),
  sample_index0 = idx - 1L,
  sample = list(
    N = proc$N[idx], E = proc$E[idx],
    qn = qn[idx], ebgm = EBGM[idx], ebgm05 = EBGM05[idx]
  )
)

out <- list(
  provenance = list(
    generated_by = "scratch script against unmodified CRAN sources",
    r_version = R.version.string,
    epitools_version = as.character(packageVersion("epitools")),
    phivid_version = "1.0.8 (sourced, not installed: LBE unavailable)",
    openebgm_version = as.character(packageVersion("openEBGM")),
    seed = 20260815
  ),
  tables = rows,
  mgps = mgps
)

write_json(out, "/ref/reference_values.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("WROTE", nrow(proc), "pairs\n")
print(theta_hat)
