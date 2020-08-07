library(acebayes)

##Code modified from acebayes source - see comment below for modification
pk_util<-function(d, B){
    D<-400
    sigadd<-0.1
    sigpro<-0.0 ## Multiplicative error term set to zero
    d2<-12*(as.vector(d)+1)
    n1<-length(d2)
    sam<-cbind(rnorm(n=2*B,mean=log(0.1),sd=sqrt(0.05)),rnorm(n=2*B,mean=log(1),sd=sqrt(0.05)),rnorm(n=2*B,mean=log(20),sd=sqrt(0.05)))
    sam<-exp(sam)
    mu<-(D/matrix(rep(sam[,3],n1),ncol=n1))*(matrix(rep(sam[,2],n1),ncol=n1)/(matrix(rep(sam[,2],n1),ncol=n1)-matrix(rep(sam[,1],n1),ncol=n1)))*(exp(-matrix(rep(sam[,1],n1),ncol=n1)*matrix(rep(d2,2*B),ncol=n1,byrow=TRUE))-exp(-matrix(rep(sam[,2],n1),ncol=n1)*matrix(rep(d2,2*B),ncol=n1,byrow=TRUE)))
    vv<-sigadd+sigpro*(mu^2)
    y<-matrix(rnorm(n=n1*B,mean=as.vector(mu[1:B,]),sd=sqrt(vv[1:B,])),ncol=n1)

    frho<-as.vector(.Call("rowSumscpp", log(vv[-(1:B),]), PACKAGE = "acebayes"))
    loglik<-as.vector(.Call("rowSumscpp", matrix(dnorm(x=as.vector(y),mean=as.vector(mu[1:B,]),sd=sqrt(vv[1:B,]),log=TRUE),ncol=n1), PACKAGE = "acebayes"))

    rsll4<-as.vector(.Call("utilcomp15sigcpp", y, mu[-(1:B),], vv[-(1:B),], frho, PACKAGE = "acebayes"))
    MY3<-log(rsll4/B)

    eval<-loglik-MY3

    eval
}

LIM.FUNC = function(i, j, d){
  # Generate a grid of 10000 points
  c(-1, sort(runif(9998))*2-1, 1)
}

### SIMULATION STUDY  ###
##library(readr)
##initial_states <- as.matrix(read_csv("~/Documents/Python/FIG/initial_states.csv", col_names = FALSE))
initial_states = runif(15,-1,1) # Times are scaled from [0,24] to [-1,1]
initial_states = matrix(initial_states, nrow=1)

ace_out = ace(utility = pk_util,
              start.d = initial_states,
              N1 = 20,
              N2 = 10,
              limits = LIM.FUNC,
              progress = TRUE)

ace_out$phase1.trace ## Utilities
12 * (sort(ace_out$phase1.d) + 1) ## Design after phase 1
12 * (sort(ace_out$phase2.d) + 1) ## Design after phase 2

##OUTPUT BELOW

## 44:02 minutes

##> ace_out$phase1.trace ## Utilities
## [1] 5.397719 5.377319 5.376047 6.381220 6.379366 6.458853 6.331691      Inf
## [9]      Inf      Inf 6.380607 6.391928 6.354861 6.347133      Inf 6.359931
##[17] 6.410248 6.417931 6.370174      Inf 6.370644
##> 12 * (sort(ace_out$phase1.d) + 1) ## Design after phase 1
## [1]  2.033192  7.011619  7.053216  7.519382  8.512532  9.081435  9.084031
## [8] 10.551505 11.551675 12.596064 13.677109 14.065566 16.630420 17.380348
##[15] 20.003635
##> 12 * (sort(ace_out$phase2.d) + 1) ## Design after phase 2
## [1]  2.033192  7.011619  7.053216  7.519382  8.512532  9.081435  9.084031
## [8] 10.551505 11.551675 12.596064 13.677109 14.065566 16.630420 17.380348
##[15] 20.003635
