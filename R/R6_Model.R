#' @title R6 class representing a (generic) model.
#'
#' @description
#' "Model" class, containing a (generic) learning function, which from
#' data + target [+ params] returns a prediction function X --> y.
#' Parameters for cross-validation are either provided or estimated.
#' Model family can be chosen among "rf", "tree", "ppr" and "knn" for now.
#'
#' @importFrom FNN knn.reg
#' @importFrom class knn
#' @importFrom stats ppr
#' @importFrom randomForest randomForest
#' @importFrom rpart rpart
#' @importFrom caret var_seq
#'
#' @export
Model <- R6::R6Class("Model",
  public = list(
    #' @field nmodels Number of parameters (= number of [predictive] models)
    nmodels = NA,
    #' @description Create a new generic model.
    #' @param data Matrix or data.frame
    #' @param target Vector of targets (generally numeric or factor)
    #' @param task "regression" or "classification"
    #' @param gmodel Generic model returning a predictive function; chosen
    #'               automatically given data and target nature if not provided.
    #' @param params List of parameters for cross-validation (each defining a model)
    initialize = function(data, target, task, gmodel = NULL, params = NULL) {
      if (is.null(gmodel)) {
        # (Generic) model not provided
        all_numeric <- is.numeric(as.matrix(data))
        if (!all_numeric)
          # At least one non-numeric column: use random forests or trees
          # TODO: 4 = arbitrary magic number...
          gmodel = ifelse(ncol(data) >= 4, "rf", "tree")
        else
          # Numerical data
          gmodel = ifelse(task == "regression", "ppr", "knn")
      }
      if (is.null(params))
        # Here, gmodel is a string (= its family),
        # because a custom model must be given with its parameters.
        params <- as.list(private$getParams(gmodel, data, target))
      private$params <- params
      if (is.character(gmodel))
        gmodel <- private$getGmodel(gmodel, task)
      private$gmodel <- gmodel
      self$nmodels <- length(private$params)
    },
    #' @description
    #' Returns the model at index "index", trained on dataHO/targetHO.
    #' index is between 1 and self$nmodels.
    #' @param dataHO Matrix or data.frame
    #' @param targetHO Vector of targets (generally numeric or factor)
    #' @param index Index of the model in 1...nmodels
    get = function(dataHO, targetHO, index) {
      private$gmodel(dataHO, targetHO, private$params[[index]])
    }
  ),
  private = list(
    # No need to expose model or parameters list
    gmodel = NULL,
    params = NULL,
    # Main function: given a family, return a generic model, which in turn
    # will output a predictive model from data + target + params.
    getGmodel = function(family, task) {
      if (family == "tree") {
        function(dataHO, targetHO, param) {
          require(rpart)
          method <- ifelse(task == "classification", "class", "anova")
          df <- data.frame(cbind(dataHO, target=targetHO))
          model <- rpart::rpart(target ~ ., df, method=method, control=list(cp=param))
          function(X) predict(model, X)
        }
      }
      else if (family == "rf") {
        function(dataHO, targetHO, param) {
          require(randomForest)
          if (task == "classification" && !is.factor(targetHO))
            targetHO <- as.factor(targetHO)
          model <- randomForest::randomForest(dataHO, targetHO, mtry=param)
          function(X) predict(model, X)
        }
      }
      else if (family == "ppr") {
        function(dataHO, targetHO, param) {
          model <- stats::ppr(dataHO, targetHO, nterms=param)
          function(X) predict(model, X)
        }
      }
      else if (family == "knn") {
        if (task == "classification") {
          function(dataHO, targetHO, param) {
            require(class)
            function(X) class::knn(dataHO, X, cl=targetHO, k=param)
          }
        }
        else {
          function(dataHO, targetHO, param) {
            require(FNN)
            function(X) FNN::knn.reg(dataHO, X, y=targetHO, k=param)$pred
          }
        }
      }
    },
    # Return a default list of parameters, given a gmodel family
    getParams = function(family, data, target) {
      if (family == "tree") {
        # Run rpart once to obtain a CV grid for parameter cp
        require(rpart)
        df <- data.frame(cbind(data, target=target))
        ctrl <- list(
          minsplit = 2,
          minbucket = 1,
          maxcompete = 0,
          maxsurrogate = 0,
          usesurrogate = 0,
          xval = 0,
          surrogatestyle = 0,
          maxdepth = 30)
        r <- rpart(target ~ ., df, method="class", control=ctrl)
        cps <- r$cptable[-1,1]
        if (length(cps) <= 11)
          return (cps)
        step <- (length(cps) - 1) / 10
        cps[unique(round(seq(1, length(cps), step)))]
      }
      else if (family == "rf") {
        p <- ncol(data)
        # Use caret package to obtain the CV grid of mtry values
        require(caret)
        caret::var_seq(p, classification = (task == "classificaton"),
                       len = min(10, p-1))
      }
      else if (family == "ppr")
        # This is nterms in ppr() function
        1:10
      else if (family == "knn") {
        n <- nrow(data)
        # Choose ~10 NN values
        K <- length(unique(target))
        if (n <= 10)
          return (1:(n-1))
        sqrt_n <- sqrt(n)
        step <- (2*sqrt_n - 1) / 10
        grid <- unique(round(seq(1, 2*sqrt_n, step)))
        if (K == 2) {
          # Common binary classification case: odd number of neighbors
          for (i in 2:11) {
            if (grid[i] %% 2 == 0)
              grid[i] <- grid[i] + 1 #arbitrary choice
          }
        }
        grid
      }
    }
  )
)
