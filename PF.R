#=====================================================================================#
# Kalman Filter and Particle Filter
#
# References (in Japanese)
#  矢野（2014）"粒子フィルタの基礎と応用：フィルタ・平滑化・パラメータ推定" 日本統計学会誌，第44巻，第1号
#  樋口（2011）『予測にいかす統計モデリングの基本』 講談社
#=====================================================================================#

#Load "dlm" package
library(dlm)

# Random seed
set.seed(1)

#======================================================#
# Data Generation
#======================================================#
# Model: Linear Gaussian State Space Model
# System model: x_{t} = x_{t-1} + v_{t}, v_{t} ~ N(0, 1)
# Observation model: y_{t} = x_{t} + ε_{t}, ε_{t} ~ N(0, 2)

T = 100
x0 = 0

# Parameters
mean_v = 0
sigma_v = sqrt(1)
mean_ep = 0
sigma_ep = sqrt(4)

# Variables
x_t_true = numeric(T)
y_t = numeric(T)

for(t in 1:T){

	x_t_true[t] = x0 + rnorm(1, mean = mean_v, sd = sigma_v)
	y_t[t] = x_t_true[t] + rnorm(1, mean = mean_ep, sd = sigma_ep)

	x0 = x_t_true[t]
}

# Plot Data
#plot(y_t, type = "l", col = 8, xlab = "Time", ylab = "")

#======================================================#
# Particle-Filter
#======================================================#

# Number of particles
M = 10000

# Number of Lags
Lag = 2

# Variables
x0_pf = numeric(M) + x_t_true[1]
pf = matrix(0, nrow = M, ncol = T)
weight = matrix(0, nrow = M, ncol = T)
pf_re = matrix(0, nrow = M, ncol = T)
x_t_pf = numeric(T)
pf_sm = matrix(0, nrow = M, ncol = T)
x_t_sm = numeric(T)

# Particle-Filter
for(t in 1:T){

	# Prediction
	pf[, t] = x0_pf + rnorm(M, mean = mean_v, sd = sigma_v)

	# Weight (=Likelihood)
	weight[, t] = (1 / sqrt(2 * pi * sigma_ep^2)) * exp( -(y_t[t] - pf[, t])^2 / (2 * sigma_ep^2) )

	# Resampling (Filtering)
    pf_re[, t] = sample(pf[, t], M, replace = TRUE, prob = weight[, t]/sum(weight[, t]))
	x0_pf = pf_re[, t]

    # Fixed-Lag Smoothing
    if(t >= Lag){
        for(k in 1:Lag){
            pf_sm[, t-k+1] = sample(pf_re[, t-k+1], M, replace = TRUE, prob = weight[, t]/sum(weight[, t]))
        }
    }
}

# State Estimate　(After Filtering)
x_t_pf = colMeans(pf_re)

# State Estimate　(After Smoothing)
x_t_sm = colMeans(pf_sm)

#======================================================#
# Kalman-Filter
#======================================================#
DLMModel = dlmModPoly(order = 1, dV = sigma_ep^2, dW = sigma_v^2)
DLMFiltering = dlmFilter(y_t, DLMModel)
DLMSmoothing = dlmSmooth(DLMFiltering)

#======================================================#
# Result
#======================================================#
par(mfrow = c(1,2))

# Filtering
plot(y_t, type = "l", col = 8, xlab = "Time", ylab = "", main = "Filtering")
lines(dropFirst(DLMFiltering$m), col = 2, lwd = 2)
lines(x_t_pf, col = 5, lwd = 2)
legend("bottomright", c("Observation", "Kalman-Filter", "Particle-Filter"), col = c(8, 2, 5), lty = 1, bty = "n")

# Smoothing
plot(y_t, type = "l", col = 8, xlab = "Time", ylab = "", main = "Smoothing")
lines(dropFirst(DLMSmoothing$s), col = 2, lwd = 2)
lines(x_t_sm, col = 5, lwd = 2)
legend("bottomright", c("Observation", "Kalman-Filter", "Particle-Filter"), col = c(8, 2, 5), lty = 1, bty = "n")
