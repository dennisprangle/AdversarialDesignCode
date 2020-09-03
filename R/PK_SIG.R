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
nreps = 30
times = vector(mode="list", length=nreps)
traces = vector(mode="list", length=nreps)
design_phase1 = vector(mode="list", length=nreps)
design_phase2 = vector(mode="list", length=nreps)
for (i in 1:nreps) {
    cat("Iteration ", i, "\n")
    set.seed(i)
    initial_states = runif(15,-1,1) # Times will be scaled from [-1,1] to [0,24]
    initial_states = matrix(initial_states, nrow=1)
    start_time = Sys.time()
    ace_out = ace(utility = pk_util,
                  start.d = initial_states,
                  N1 = 20,
                  N2 = 10,
                  limits = LIM.FUNC,
                  progress = TRUE)
    times[[i]] = Sys.time() - start_time
    traces[[i]] = ace_out$phase1.trace
    design_phase1[[i]] = 12 * (sort(ace_out$phase1.d) + 1)
    design_phase2[[i]] = 12 * (sort(ace_out$phase2.d) + 1)
}
save(times, traces, design_phase1, design_phase2, file="../outputs/pk.Rdata")
