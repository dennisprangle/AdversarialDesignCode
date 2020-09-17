library(acebayes)
library(Rcpp)

sourceCpp("utilcomp.cpp")

##Code modified from acebayes source - see comment below for modification
pk_util<-function(d, B){
    D<-400
    sigadd<-0.1
    d2<-12*(as.vector(d)+1)
    n1<-length(d2)
    sam<-cbind(rnorm(n=2*B,mean=log(0.1),sd=sqrt(0.05)),rnorm(n=2*B,mean=log(1),sd=sqrt(0.05)),rnorm(n=2*B,mean=log(20),sd=sqrt(0.05)))
    sam<-exp(sam)
    mu<-(D/matrix(rep(sam[,3],n1),ncol=n1))*(matrix(rep(sam[,2],n1),ncol=n1)/(matrix(rep(sam[,2],n1),ncol=n1)-matrix(rep(sam[,1],n1),ncol=n1)))*(exp(-matrix(rep(sam[,1],n1),ncol=n1)*matrix(rep(d2,2*B),ncol=n1,byrow=TRUE))-exp(-matrix(rep(sam[,2],n1),ncol=n1)*matrix(rep(d2,2*B),ncol=n1,byrow=TRUE)))
    vv<-sigadd + 0*mu ## **MODIFICATION** - remove multiplicative error term
    y<-matrix(rnorm(n=n1*B,mean=as.vector(mu[1:B,]),sd=sqrt(vv[1:B,])),ncol=n1)

    frho<-as.vector(.Call("rowSumscpp", log(vv[-(1:B),]), PACKAGE = "acebayes"))
    loglik<-as.vector(.Call("rowSumscpp", matrix(dnorm(x=as.vector(y),mean=as.vector(mu[1:B,]),sd=sqrt(vv[1:B,]),log=TRUE),ncol=n1), PACKAGE = "acebayes"))

    MY3 <- utilcomp15sigcpp_edit(y, mu[-(1:B),], vv[-(1:B),], frho) ## **MODIFICATION** - cpp code editted to make log mean exp more numerically stable
    eval<-loglik-MY3
    if( any(!is.finite(eval)) ) stop("Infinite or NaN utility estimate") ## **MODIFICATION** - check no overflow/underflow issues
    eval
}

### SIMULATION STUDY  ###
nreps = 30
times = vector(mode="list", length=nreps)
traces = vector(mode="list", length=nreps)
design_out = vector(mode="list", length=nreps)
for (i in 1:nreps) {
    cat("Iteration ", i, "\n")
    set.seed(i)
    initial_states = runif(15,-1,1) # Times will be scaled from [-1,1] to [0,24]
    initial_states = matrix(initial_states, nrow=1)
    start_time = Sys.time()
    ace_out = ace(utility = pk_util,
                  start.d = initial_states,
                  N1 = 20,
                  N2 = 0,
                  progress = TRUE)
    times[[i]] = difftime(Sys.time(), start_time, units="secs")
    traces[[i]] = ace_out$phase1.trace
    design_out[[i]] = 12 * (sort(ace_out$phase1.d) + 1)
}

## Convert results into a single data frame and save as csv
times = unlist(times)
traces = do.call(rbind, traces)
colnames(traces) = paste0("traces_", 1:ncol(traces))
design_out = do.call(rbind, design_out)
colnames(design_out) = paste0("design_", 1:ncol(design_out))
results = cbind(times, traces, design_out)

write.csv(results, file="../outputs/pk_SIG.csv", row.names=FALSE)
