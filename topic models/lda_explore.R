rm(list=ls())

library(topicmodels)
library(lda)
library(rjson)
library(gplots)
library(foreach)

getDocs <- function(x){
  documents <- list()
  for(i in 1:length(x)){
    doc <- strsplit(x[i], ' ')[[1]]
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
  return(documents)
}

getMbyD <- function(x,wc){
  res <- list()
  for(i in 1:length(x)){
    item <- x[i]
    item <- strsplit(item, ',')[[1]]
    doc <- strsplit(wc[i], ' ')[[1]]
    if(strtoi(doc[1]) > 0) {
      res <- c(res, list(item))
    }
  }
  return(res)
}

#######################################################################################
#######################################################################################
# ENV SETUP
pref_path <- getwd() # /home/<user>/
count_pardir <- paste(pref_path,'/empanadas/dataR/counts/',sep='')
count_dir <- ''
#

#######################################################################################
## set which data to use
NE <- 0 # 0: NE_sp | 1: NE_sp_hmm | 2: FULL

if(NE==0){
  ne_counts <- paste(count_pardir,'NE_ene_ing/',sep='')
  ne_wc   <- scan(paste(ne_counts  ,'word_counts.dat',sep=''), what='', sep = '\n')
  ne_documents   <- getDocs(ne_wc)
  ne_vocab   <- scan(paste(ne_counts  ,'vocab.dat',sep=''), what='', sep = '\n')
  ne_major_by_doc   <- scan(paste(ne_counts  ,'majors_by_doc.dat',sep=''), what='',sep = '\n')
  ne_major_by_doc   <- getMbyD(ne_major_by_doc,ne_wc)
  
  ne_title_map <- scan(paste(ne_counts  ,'title_map.dat',sep=''), what='', sep = '\n')
  
  pref <-'ne'
  major_by_doc <- ne_major_by_doc
  documents <- ne_documents
  vocab <- ne_vocab
  title_map <- ne_title_map
  count_dir <- ne_counts
  
}else if(NE==2){
  full_counts <- paste(count_pardir,'full_ene_ing/',sep='')
  full_wc <- scan(paste(full_counts,'word_counts.dat',sep=''), what='', sep = '\n')
  full_documents <- getDocs(full_wc)
  full_vocab <- scan(paste(full_counts,'vocab.dat',sep=''), what='', sep = '\n')
  full_major_by_doc <- scan(paste(full_counts  ,'majors_by_doc.dat',sep=''), what='',sep = '\n')
  full_major_by_doc <- getMbyD(full_major_by_doc,full_wc)
  
  pref <- 'full'
  major_by_doc <- full_major_by_doc
  documents <- full_documents
  vocab <- full_vocab
}else{
  ne_counts <- paste(count_pardir,'NE_ene_ing_sp_hmm/',sep='')
  ne_wc   <- scan(paste(ne_counts  ,'word_counts.dat',sep=''), what='', sep = '\n')
  ne_documents   <- getDocs(ne_wc)
  ne_vocab   <- scan(paste(ne_counts  ,'vocab.dat',sep=''), what='', sep = '\n')
  ne_major_by_doc   <- scan(paste(ne_counts  ,'majors_by_doc.dat',sep=''), what='',sep = '\n')
  ne_major_by_doc   <- getMbyD(ne_major_by_doc,ne_wc)
  
  pref <-'ne_hmm'
  major_by_doc <- ne_major_by_doc
  documents <- ne_documents
  vocab <- ne_vocab
}

#######################################################################################
THR = 45
major_map    <- fromJSON(file=paste(count_pardir,'map_major_eng.json',sep=''))
major_counts <- fromJSON(file=paste(count_pardir,'major_counts_eng.json',sep=''))

filtered_docs = list()
filtered_major_by_doc = list()
filtered_title_map = list()

index_filter_mc = sapply(major_counts,function(x) return(x>THR))
filtered_major_counts = major_counts[index_filter_mc]

n_majors     <- length(filtered_major_counts)
major_names  <- names(filtered_major_counts)

#reset counts for every major
major_counts <- lapply(major_counts,function(x) return(0))

# count majors
for (i in 1:length(documents)){
  foreach(major=major_by_doc[[i]])%do% {
    cc = major_counts[[ major_map[[major]] ]]
    major_counts[[ major_map[[major]] ]] = cc + 1
  }
}
filtered_major_counts = major_counts[index_filter_mc]
## filter by freq_major
p=1
for (i in 1:length(documents)){
  high_freq = FALSE
  major_list = c()
  foreach(major=major_by_doc[[i]])%do% {
    if(major_counts[[major_map[[major]] ]]>THR){
      high_freq = TRUE
      major_list = c(major_list,major)
    }
  }
  if(high_freq){
    filtered_docs = c(filtered_docs,documents[i])
    filtered_major_by_doc[[p]] = major_list
    filtered_title_map = c(filtered_title_map,title_map[i])
    p=p+1
  }
}

# Write filtered title_map data
write.table(filtered_title_map, paste(count_dir,"filtered_title_map.dat",sep='/'), 
            sep="\n",row.names=FALSE,col.names=FALSE, quote=FALSE)

#######################################################################################

data   <- ldaformat2dtm(filtered_docs , vocab  , omit_empty = TRUE)

#######################################################################################
#######################################################################################
fixed_alpha <- 0.06
K <- 26
SEED <- 42
if(NE==1){ # NE_HMM
  control_LDA_VEM_est <- list(estimate.alpha = TRUE, estimate.beta = TRUE,
                            verbose = 10, prefix = tempfile(), save = 0, keep = 5,
                            seed = SEED, nstart = 1, best = TRUE,
                            var = list(iter.max = 2000, tol = 10^-5), #10^-5 para NE_hmm
                            em = list(iter.max = 2000, tol = 10^-5), #10^-5 para NE_hmm
                            initialize = "random")
}else{
  control_LDA_VEM_est <- list(estimate.alpha = TRUE, estimate.beta = TRUE,
                              verbose = 10, prefix = tempfile(), save = 0, keep = 5,
                              seed = SEED, nstart = 1, best = TRUE,
                              var = list(iter.max = 1000, tol = 10^-6), 
                              em = list(iter.max = 1000, tol = 10^-4),
                              initialize = "random")
}

control_LDA_VEM_fix <- list(estimate.alpha = FALSE, alpha=fixed_alpha,estimate.beta = TRUE,
                        verbose = 10, prefix = tempfile(), save = 0, keep = 5,
                        seed = SEED, nstart = 1, best = TRUE,
                        var = list(iter.max = 1000, tol = 10^-6),
                        em = list(iter.max = 1000, tol = 10^-4),
                        initialize = "random")

control_LDA_GIBBS <- list(alpha = 5/K, estimate.beta = TRUE,
                          verbose = 10, prefix = tempfile(), save = 0, keep = 5,
                          seed = SEED, nstart = 1, best = TRUE,
                          delta = 0.1, iter = 2000, burnin = 0, thin = 2000)

control_CTM_VEM <- list(estimate.beta = TRUE, verbose = 0, prefix = tempfile(), save = 0,
                     keep = 0, seed = SEED, nstart = 1L, best = TRUE,
                     var = list(iter.max = 500, tol = 10^-6), em = list(iter.max = 1000, tol = 10^-4),
                     initialize = "random", cg = list(iter.max = 500, tol = 10^-5))


#######################################################################################
TM_list <- list(
  VEM_est = LDA(x=data, control=control_LDA_VEM_est, k=K, method="VEM"),
  VEM_fix = LDA(x=data, control=control_LDA_VEM_fix, k=K, method="VEM"),
  Gibbs   = LDA(x=data, control=control_LDA_GIBBS  , k=K, method="Gibbs")
  #,CTM_VEM = CTM(x = ne_data, control = control_CTM_VEM , k = K, method = "VEM")
)

#######################################################################################
# major vs topic

major_vs_topic <- lapply(TM_list,
                         function(x){
                           tp <- posterior(x)$topics
                           maj_top <- matrix(0,n_majors,dim(tp)[2],dimnames=list(major_names))
                           for(i in 1:dim(tp)[1]){
                             for(j in 1:length(filtered_major_by_doc[[i]])){
                               major <- major_map[[ filtered_major_by_doc[[i]][j] ]]
                               maj_top[major,] <- maj_top[major,]+tp[i,]
                             }
                           }
                           for(i in 1:n_majors){
                             major <- major_names[i]
                             maj_top[major,] <- maj_top[major,]/filtered_major_counts[[major]]
                           }
                           return(maj_top)
                         })


#######################################################################################
## check alphas
sapply(TM_list, slot, "alpha")

# check entropy measure
ent <- sapply(TM_list, function(x)
          mean(apply(posterior(x)$topics,
                      1, function(z) - sum(z * log(z)))))

#ne_topics <- lapply(TM_list,function(x) posterior(x)$topics)


#######################################################################################
# Most likely topic histogram

mostliktopic <- sapply(TM_list,function(x)
                                  apply(posterior(x)$topics,1,function(y) return(max(y)) )
                       )

# plot histograms

h1 = hist(mostliktopic[,"VEM_est"])
h1$density = 100*h1$counts/sum(h1$counts)

h2 = hist(mostliktopic[,"VEM_fix"])
h2$density = 100*h2$counts/sum(h2$counts)

h3 = hist(mostliktopic[,"Gibbs"])
h3$density = 100*h3$counts/sum(h3$counts)

png(paste(pref,'_most_likely_topic_',toString(K),'.png',sep=''), width=900, height=500, res=200, pointsize=5)
par(mfrow=c(1,3))
plot(h1,freq=FALSE, ylab="Percentage", main="VEM alpha estimated")
plot(h2,freq=FALSE, ylab="Percentage", main="VEM alpha fixed")
plot(h3,freq=FALSE, ylab="Percentage", main="Gibbs alpha_ini=5/K")
dev.off()


#######################################################################################
major_hist = "Food"

#######################################################################################
#######################################################################################
## HEATMAPS  MAJOR vs TOPICS

PERCT_THR = min(1/K,1/n_majors) # debajo de 1/n pone a 0
MvsT_filtered <- lapply(major_vs_topic,
                        function(x){
                          min_val <- min(x)
                          max_val <- max(x)
                          THR <- min_val + (max_val-min_val)*PERCT_THR
                          res = x
                          res[res<THR] <- 0
                          return(res)
                        })
n_colorbins=5000
color_breaks = seq(0,0.7,length=n_colorbins)

png(paste(pref,'_vem_est_',toString(K),'.png',sep=''), width=900, height=600, res=600, pointsize=1.5)
heatmap.2(MvsT_filtered[["VEM_est"]],
          main="VEM estimated alpha",
          # dendogram control
          dendrogram ='none',
          Rowv = FALSE,
          Colv = FALSE,
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('white','yellow','red'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=TRUE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

png(paste(pref,'_vem_fix_',toString(K),'.png',sep=''), width=900, height=600, res=300, pointsize=2)
heatmap.2(MvsT_filtered[["VEM_fix"]],
          main="VEM fixed alpha",
          # dendogram control
          dendrogram ='none',
          Rowv = FALSE,
          Colv = FALSE,
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('white','yellow','red'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)

dev.off()

png(paste(pref,'_gibbs_',toString(K),'.png',sep=""), width=900, height=600, res=300, pointsize=2)
heatmap.2(MvsT_filtered[["Gibbs"]],
          main="Gibbs",
          # dendogram control
          dendrogram ='none',
          Rowv = FALSE,
          Colv = FALSE,
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('white','yellow','red'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

#######################################################################################
#######################################################################################
## HEATMAPS  TOPICS vs TOPICS | CORRELATION

png(paste(pref,'_TvsT_vem_est_',toString(K),'.png',sep=""), width=700, height=600, res=300, pointsize=2)
heatmap.2(cor(major_vs_topic[["VEM_est"]]),
          main="VEM alpha estimated",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'white','yellow','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,5),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

png(paste(pref,'_TvsT_vem_fix_',toString(K),'.png',sep=""), width=700, height=600, res=300, pointsize=2)
heatmap.2(cor(major_vs_topic[["VEM_fix"]]),
          main="VEM alpha fixed",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'white','yellow','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,5),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

png(paste(pref,'_TvsT_gibbs_',toString(K),'.png',sep=""), width=700, height=600, res=300, pointsize=2)
heatmap.2(cor(major_vs_topic[["Gibbs"]]),
          main="Gibbs",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'white','yellow','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(5,5),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

#######################################################################################
#######################################################################################
## HEATMAPS  MAJOR vs MAJOR | CORRELATION

png(paste(pref,'_MvsM_vem_est_',toString(K),'.png',sep=""), width=900, height=900, res=300, pointsize=2)
heatmap.2(cor(t(major_vs_topic[["VEM_est"]])),
          main="VEM alpha estimated",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'blue','green','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

png(paste(pref,'_MvsM_vem_fix_',toString(K),'.png',sep=""), width=900, height=900, res=300, pointsize=2)
heatmap.2(cor(t(major_vs_topic[["VEM_fix"]])),
          main="VEM alpha fixed",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'blue','green','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

png(paste(pref,'_MvsM_gibbs_',toString(K),'.png',sep=""), width=900, height=900, res=300, pointsize=2)
heatmap.2(cor(t(major_vs_topic[["Gibbs"]])),
          main="Gibbs",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorpanel(5000,'blue','green','red'),
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(0.2,4),
)
dev.off()

#######################################################################################
#######################################################################################
## HEATMAPS  MAJOR vs MAJOR | HELLINGER
color_breaks = seq(0,0.75,length=n_colorbins)

png(paste(pref,'_MvsM_hellinger_vem_est_',toString(K),'.png',sep=""), width=900, height=900, res=400, pointsize=2)
heatmap.2(distHellinger( major_vs_topic[["VEM_est"]],major_vs_topic[["VEM_est"]] ),
          main="VEM alpha estimated",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('red','yellow','blue'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=TRUE,
          keysize=1.5, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          labRow=major_names,
          labCol=major_names,
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(1,2),
)
dev.off()

png(paste(pref,'_MvsM_hellinger_vem_fix_',toString(K),'.png',sep=""), width=900, height=900, res=300, pointsize=2)
heatmap.2(distHellinger( major_vs_topic[["VEM_fix"]],major_vs_topic[["VEM_fix"]] ),
          main="VEM alpha fixed",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('red','yellow','blue'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          labRow=major_names,
          labCol=major_names,
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(1,2),
)
dev.off()

png(paste(pref,'_MvsM_hellinger_gibbs_',toString(K),'.png',sep=""), width=900, height=900, res=300, pointsize=2)
heatmap.2(distHellinger( major_vs_topic[["Gibbs"]],major_vs_topic[["Gibbs"]] ),
          main="Gibbs",
          # dendogram control
          dendrogram ='none',
          distfun=dist,
          hclustfun=function(c){hclust(c, method='mcquitty')},
          # data scaling         
          scale="none",
          cexRow=2,cexCol=2,
          # colors
          col=colorRampPalette(c('red','yellow','blue'))(n_colorbins-1),
          breaks=color_breaks,
          # color key + density info
          key=FALSE,
          keysize=1.0, symkey=FALSE, density.info='none',
          trace='none',
          # row/column labeling | plot layout
          labRow=major_names,
          labCol=major_names,
          margins=c(30,30),
          #lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.1, 4, 0.1 ), lwid=c(1,2),
)
dev.off()

#######################################################################################
#######################################################################################
# WRITE TERMS AND PARAMS OF EACH MODEL
terms_by_top <- 20

foreach(model=names(TM_list))%do% {
  beta <- slot(TM_list[[model]],"beta")
  gamma <- slot(TM_list[[model]],"gamma")
  write.table(beta, paste("beta_",model,toString(K),sep=''), sep=" ",row.names=FALSE,col.names=FALSE)
  write.table(gamma, paste("gamma_",model,toString(K),sep=''), sep=" ",row.names=FALSE,col.names=FALSE)
  
  mod_terms <- terms(TM_list[[model]],terms_by_top)
  write.table(mod_terms, paste("terms_",model,toString(K),'.csv',sep=''), sep=",",row.names=FALSE,col.names=TRUE)
  
  #write.table(major_vs_topic, paste("terms_",model,toString(K),'.csv',sep=''), sep=",",row.names=FALSE,col.names=TRUE)
}

