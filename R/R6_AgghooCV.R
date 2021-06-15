#' @title R6 class with agghoo functions fit() and predict().
#'
#' @description
#' Class encapsulating the methods to run to obtain the best predictor
#' from the list of models (see 'Model' class).
#'
#' @importFrom R6 R6Class
#'
#' @export
AgghooCV <- R6::R6Class("AgghooCV",
  public = list(
    #' @description Create a new AgghooCV object.
    #' @param data Matrix or data.frame
    #' @param target Vector of targets (generally numeric or factor)
    #' @param task "regression" or "classification"
    #' @param gmodel Generic model returning a predictive function
    #' @param loss Function assessing the error of a prediction
    initialize = function(data, target, task, gmodel, loss = NULL) {
      private$data <- data
      private$target <- target
      private$task <- task
      private$gmodel <- gmodel
      if (is.null(loss))
        loss <- private$defaultLoss
      private$loss <- loss
    },
    #' @description Fit an agghoo model.
    #' @param CV List describing cross-validation to run. Slots:
    #'          - type: 'vfold' or 'MC' for Monte-Carlo (default: MC)
    #'          - V: number of runs (default: 10)
    #'          - test_size: percentage of data in the test dataset, for MC
    #'            (irrelevant for V-fold). Default: 0.2.
    #'          - shuffle: wether or not to shuffle data before V-fold.
    #'            Irrelevant for Monte-Carlo; default: TRUE
    fit = function(
      CV = list(type = "MC",
                V = 10,
                test_size = 0.2,
                shuffle = TRUE)
    ) {
      if (!is.list(CV))
        stop("CV: list of type, V, [test_size], [shuffle]")
      n <- nrow(private$data)
      shuffle_inds <- NULL
      if (CV$type == "vfold" && CV$shuffle)
        shuffle_inds <- sample(n, n)
      # Result: list of V predictive models (+ parameters for info)
      private$pmodels <- list()
      for (v in seq_len(CV$V)) {
        # Prepare train / test data and target, from full dataset.
        # dataHO: "data Hold-Out" etc.
        test_indices <- private$get_testIndices(CV, v, n, shuffle_inds)
        dataHO <- private$data[-test_indices,]
        testX <- private$data[test_indices,]
        targetHO <- private$target[-test_indices]
        testY <- private$target[test_indices]
        # [HACK] R will cast 1-dim matrices into vectors:
        if (!is.matrix(dataHO) && !is.data.frame(dataHO))
          dataHO <- as.matrix(dataHO)
        if (!is.matrix(testX) && !is.data.frame(testX))
          testX <- as.matrix(testX)
        best_model <- NULL
        best_error <- Inf
        for (p in seq_len(private$gmodel$nmodels)) {
          model_pred <- private$gmodel$get(dataHO, targetHO, p)
          prediction <- model_pred(testX)
          error <- private$loss(prediction, testY)
          if (error <= best_error) {
            newModel <- list(model=model_pred, param=private$gmodel$getParam(p))
            if (error == best_error)
              best_model[[length(best_model)+1]] <- newModel
            else {
              best_model <- list(newModel)
              best_error <- error
            }
          }
        }
        # Choose a model at random in case of ex-aequos
        private$pmodels[[v]] <- best_model[[ sample(length(best_model),1) ]]
      }
    },
    #' @description Predict an agghoo model (after calling fit())
    #' @param X Matrix or data.frame to predict
    predict = function(X) {
      if (!is.matrix(X) && !is.data.frame(X))
        stop("X: matrix or data.frame")
      if (!is.list(private$pmodels)) {
        print("Please call $fit() method first")
        return (invisible(NULL))
      }
      V <- length(private$pmodels)
      oneLineX <- t(as.matrix(X[1,]))
      if (length(private$pmodels[[1]]$model(oneLineX)) >= 2)
        # Soft classification:
        return (Reduce("+", lapply(private$pmodels, function(m) m$model(X))) / V)
      n <- nrow(X)
      all_predictions <- as.data.frame(matrix(nrow=n, ncol=V))
      for (v in 1:V)
        all_predictions[,v] <- private$pmodels[[v]]$model(X)
      if (private$task == "regression")
        # Easy case: just average each row
        return (rowMeans(all_predictions))
      # "Hard" classification:
      apply(all_predictions, 1, function(row) {
        t <- table(row)
        # Next lines in case of ties (broken at random)
        tmax <- max(t)
        sample( names(t)[which(t == tmax)], 1 )
      })
    },
    #' @description Return the list of V best parameters (after calling fit())
    getParams = function() {
      lapply(private$pmodels, function(m) m$param)
    }
  ),
  private = list(
    data = NULL,
    target = NULL,
    task = NULL,
    gmodel = NULL,
    loss = NULL,
    pmodels = NULL,
    get_testIndices = function(CV, v, n, shuffle_inds) {
      if (CV$type == "vfold") {
        # Slice indices (optionnally shuffled)
        first_index = round((v-1) * n / CV$V) + 1
        last_index = round(v * n / CV$V)
        test_indices = first_index:last_index
        if (!is.null(shuffle_inds))
          test_indices <- shuffle_inds[test_indices]
      }
      else
        # Monte-Carlo cross-validation
        test_indices = sample(n, round(n * CV$test_size))
      test_indices
    },
    defaultLoss = function(y1, y2) {
      if (private$task == "classification") {
        if (is.null(dim(y1)))
          # Standard case: "hard" classification
          mean(y1 != y2)
        else {
          # "Soft" classification: predict() outputs a probability matrix
          # In this case "target" could be in matrix form.
          if (!is.null(dim(y2)))
            mean(rowSums(abs(y1 - y2)))
          else {
            # Or not: y2 is a "factor".
            y2 <- as.character(y2)
            # NOTE: the user should provide target in matrix form because
            # matching y2 with columns is rather inefficient!
            names <- colnames(y1)
            positions <- list()
            for (idx in seq_along(names))
              positions[[ names[idx] ]] <- idx
            mean(vapply(
              seq_along(y2),
              function(idx) sum(abs(y1[idx,] - positions[[ y2[idx] ]])),
              0))
          }
        }
      }
      else
        # Regression
        mean(abs(y1 - y2))
    }
  )
)
