.T_hdc_getTheModel = function(model, all2models = FALSE){
  # Function used to get the models from number or names
  
  model_in = model
  
  if(!is.vector(model)) stop("The argument 'model' must be a vector.")
  
  if(anyNA(model)) stop("The argument 'model' must not contain any NA.")
  
  ModelNames <- c("AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD")
  
  model = toupper(model)
  
  if(length(model)==1 && model=="ALL"){
    if(all2models) model <- 1:14
    else return("ALL")
  }
  
  qui = which(model %in% 1:14)
  model[qui] = ModelNames[as.numeric(model[qui])]
  
  # We order the models properly
  qui = which(!model%in%ModelNames)
  if (length(qui)>0){
    if(length(qui)==1){
      msg = paste0("(e.g. ", model_in[qui], " is incorrect.)")
    } else {
      msg = paste0("(e.g. ", paste0(model_in[qui[1:2]], collapse=", or "), " are incorrect.)")
    }
    stop("Invalid model name ", msg)
  }
  
  # warning:
  if(max(table(model))>1) warning("The model vector, argument 'model', is made unique (repeated values are not tolerated).")
  
  mod_num <- c()
  for(i in 1:length(model)) mod_num[i] <- which(model[i]==ModelNames)
  mod_num <- sort(unique(mod_num))
  model <- ModelNames[mod_num]
  
  return(model)
}

a = c("akjbkqkdk", 'akbkqkdk')
b = c('akbkqkdk', 'akjbkqkdk')
c = c('akbkqkdk', 'akbqkdk', 'akjbkqkdk', 'AKBKQKD')
d = c('1', '2', '3')
e = c(1, 2, 3)
f = c(3, 4, 2, 1)
g = c(1, 15, 16, 17)
h = c('akbkqkdk', 'akjbkqkdk', 'a')
i = c('akjbkqkdk', 'akjbkqkdk')
j = c('2', '2')

tests = list(a = a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i, j=j)

for(i in tests){
  try(print(.T_hdc_getTheModel(i)))
  
}