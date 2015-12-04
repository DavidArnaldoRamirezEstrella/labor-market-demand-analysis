library(topicmodels)
library(lda)


#x <- scan('DT_DATA_R.dat', what='', sep = '\n')
x <- scan('word_counts.dat', what='', sep = '\n')
#x <- strsplit(x, ' ')
#x <- lapply(x, strtoi)

documents <- list()
for(i in 1:length(x)){
  #documents <- c( documents, list( matrix(x[[i]],2) ) )
  doc <- x[i]
  doc <- strsplit(doc, ' ')
  doc <- doc[[1]]
  if(strtoi(doc[1]) > 0) {
      matrix_temp <- matrix(1:(strtoi(doc[1])*2), 2)
      for(j in 2:length(doc)){
        ind_freq <- doc[j]
        ind_freq <- strsplit(ind_freq, ':')
        ind_freq <- ind_freq[[1]] 
        matrix_temp[1, j-1] <- strtoi(ind_freq[1])
        matrix_temp[2, j-1] <- strtoi(ind_freq[2])
      }
      documents <- c( documents, list( matrix_temp ) )
  }
}

mm <- length(documents)

#vocab <- scan('VOCAB_DATA_R.dat', what='', sep = '\n')
vocab <- scan('vocab.dat', what='', sep = '\n')

## SPLINT IN KV FOLDS
KV <- 10
folds <- list()
if( mm %% KV == 0) {
  width_fold <- mm / KV
  ini <- 1
  for ( ii in 1:KV ) {
    folds[[ii]] <- documents[ini:(ini+width_fold-1)]
    ini <- ini + width_fold
  }
} else{
  width_fold <- floor(mm / KV) + 1
  ini <- 1
  for ( ii in 1:(mm %% KV) ) {
    folds[[ii]] <- documents[ini:(ini+width_fold-1)]
    ini <- ini + width_fold
  }  
  width_fold <- floor(mm / KV)
  for ( ii in (mm %% KV + 1):KV ) {
    folds[[ii]] <- documents[ini:(ini+width_fold-1)]
    ini <- ini + width_fold
  }
}


perplex <- 10000000000
optimo <- 0
resp <- c()
resp2 <- c()
loglik <- c()

alphas <- c(1e-5,1e-4,1e-3,0.01,0.05,0.08,0.1,0.5,0.8,1,1.5,2,5,10)
#alphas <- c(0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.6,0.7)
K <- 26
parallelize <- TRUE

if (parallelize){
  library(foreach)
  library(doParallel)
  no_cores <- detectCores()
  cl<-makeCluster(no_cores)
  registerDoParallel(cl)
}


strt <- Sys.time()


LDA_method <- 0  # 0 = VEM, 1 = Gibbs

for(alpha in alphas){
  control_LDA_VEM <- list(estimate.alpha = FALSE, alpha=alpha,estimate.beta = TRUE,
                     #verbose = 4, prefix = 'iter', save = 4, keep = 4,
                     verbose = 10, prefix = tempfile(), save = 0, keep = 0,
                     seed = as.integer(Sys.time()), nstart = 1, best = TRUE,
                     var = list(iter.max = 1000, tol = 10^-6),
                     em = list(iter.max = 1000, tol = 10^-4),
                     initialize = "random")

  control_LDA_GIBBS <- list(estimate.alpha = FALSE,alpha = alpha, estimate.beta = TRUE,
                       verbose = 10, prefix = tempfile(), save = 0, keep = 0,
                       seed = as.integer(Sys.time()), nstart = 1, best = TRUE,
                       delta = 0.1, iter = 2000, burnin = 0, thin = 2000)

  perplex_model <- 0
  perplex_test <- 0
  loglog <- 0
  min_per <- 100000000
  if (parallelize){
    compact <- foreach ( ii=1:KV ) %dopar% {
                library(topicmodels)
                TRAIN <- list()
                TEST <- folds[[ii]]
                for( jj in 1:KV ) {
                  if ( jj != ii ) {
                    TRAIN <- c(TRAIN, folds[[jj]])
                  }
                }
                train_docs <- ldaformat2dtm(TRAIN, vocab, omit_empty = TRUE)
                test_docs <- ldaformat2dtm(TEST, vocab, omit_empty = TRUE)

                if(LDA_method){
                    lda_model <- LDA(x = train_docs, control = control_LDA_GIBBS , k = K, method = "Gibbs")
                } else{
                    lda_model <- LDA(x = train_docs, control = control_LDA_VEM , k = K, method = "VEM")
                }

                if(LDA_method == 0){
                    par_perplex_model <- perplexity(lda_model)
                }else{
                    par_perplex_model <- perplexity(lda_model, test_docs)
                }
                par_perplex_test <- perplexity(lda_model, test_docs)
                par_loglog <- logLik(lda_model)
                c(par_perplex_model,par_perplex_test,par_loglog)
              }
    results <- matrix(unlist(compact),nrow=KV,ncol=3,byrow=TRUE)
    min_per <- min(results[,2])
    print(" ** Minimun perplexity of hold-out data during Cross-Validation **")
    print(min_per)
    perplex_model <- sum(results[,1]) / KV
    perplex_test <- sum(results[,2]) / KV
    loglog <- sum(results[,3]) / KV
  }else{
    for ( ii in 1:KV ) {
      print("INFO UPDATED>>>>>>>>> alpha - #fold")
      print(alpha)
      print(ii)
      TRAIN <- list()
      TEST <- folds[[ii]]
      for( jj in 1:KV ) {
        if ( jj != ii ) {
          TRAIN <- c(TRAIN, folds[[jj]])
        }
      }
      train_docs <- ldaformat2dtm(TRAIN, vocab, omit_empty = TRUE)
      test_docs <- ldaformat2dtm(TEST, vocab, omit_empty = TRUE)
      
      if(LDA_method){
          lda_model <- LDA(x = train_docs, control = control_LDA_GIBBS , k = K, method = "Gibbs")
      } else{
          lda_model <- LDA(x = train_docs, control = control_LDA_VEM , k = K, method = "VEM")
      }
      if(LDA_method == 0){
        perplex_model <- perplex_model + perplexity(lda_model)
      } else{
        perplex_model <- perplex_model + perplexity(lda_model, test_docs)
      }
      perplex_test <- perplex_test + perplexity(lda_model, test_docs)
      loglog <- loglog + logLik(lda_model)
      min_per <- min(min_per, perplexity(lda_model, test_docs))
    }
    print(" ** Minimun perplexity of hold-out data during Cross-Validation **")
    print(min_per)
    perplex_model <- perplex_model / KV
    perplex_test <- perplex_test / KV
    loglog <- loglog / KV
  }
  metric <- perplex_test
  resp <- c(resp, metric)
  resp2 <- c(resp2, perplex_model)
  loglik <- c(loglik, loglog)
  if(metric < perplex){
    perplex <-metric
    optimo <- K
    print(" << Optimal values so far >> ")
    print(optimo)
    print(perplex)
  }
}
print(Sys.time()-strt)
  
print(resp)
stopCluster(cl)
#terms(lda_model, 5)
#lda_model <- LDA(x = train_docs, control = control_LDA_VEM , k = K, method = "VEM")
#terms(lda_model, 5)
#perplexity(lda_model)
#perplexity(lda_model, test_docs)
#lda2_model <- lda.collapsed.gibbs.sampler(documents, 70, vocab, 150, 1, 0.01)

#predictions <- predictive.distribution(lda2_model$document_sums[,1:2], lda2_model$topics, 1, 0.01)
#top.topic.words(lda2_model$topics, 10,by.score = FALSE)
