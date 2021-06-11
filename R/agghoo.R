#' agghoo
#'
#' Run the agghoo procedure (or standard cross-validation).
#' Arguments specify the list of models, their parameters and the
#' cross-validation settings, among others.
#'
#' @param data Data frame or matrix containing the data in lines.
#' @param target The target values to predict. Generally a vector,
#'        but possibly a matrix in the case of "soft classification".
#' @param task "classification" or "regression". Default:
#'        regression if target is numerical, classification otherwise.
#' @param gmodel A "generic model", which is a function returning a predict
#'        function (taking X as only argument) from the tuple
#'        (dataHO, targetHO, param), where 'HO' stands for 'Hold-Out',
#'        referring to cross-validation. Cross-validation is run on an array
#'        of 'param's. See params argument. Default: see R6::Model.
#' @param params A list of parameters. Often, one list cell is just a
#'        numerical value, but in general it could be of any type.
#'        Default: see R6::Model.
#' @param loss A function assessing the error of a prediction.
#'        Arguments are y1 and y2 (comparing a prediction to known values).
#'        loss(y1, y2) --> real number (error). Default: see R6::AgghooCV.
#'
#' @return
#' An R6::AgghooCV object o. Then, call o$fit() and finally o$predict(newData)
#'
#' @examples
#' # Regression:
#' a_reg <- agghoo(iris[,-c(2,5)], iris[,2])
#' a_reg$fit()
#' pr <- a_reg$predict(iris[,-c(2,5)] + rnorm(450, sd=0.1))
#' # Classification
#' a_cla <- agghoo(iris[,-5], iris[,5])
#' a_cla$fit()
#' pc <- a_cla$predict(iris[,-5] + rnorm(600, sd=0.1))
#'
#' @references
#' Guillaume Maillard, Sylvain Arlot, Matthieu Lerasle. "Aggregated hold-out".
#' Journal of Machine Learning Research 22(20):1--55, 2021.
#'
#' @export
agghoo <- function(data, target, task = NULL, gmodel = NULL, params = NULL, loss = NULL) {
	# Args check:
  if (!is.data.frame(data) && !is.matrix(data))
    stop("data: data.frame or matrix")
  if (is.data.frame(target) || is.matrix(target)) {
    if (nrow(target) != nrow(data) || ncol(target) == 1)
      stop("target probability matrix does not match data size")
  }
  else if (!is.numeric(target) && !is.factor(target) && !is.character(target))
    stop("target: numeric, factor or character vector")
  if (!is.null(task))
    task = match.arg(task, c("classification", "regression"))
  if (is.character(gmodel))
    gmodel <- match.arg(gmodel, c("knn", "ppr", "rf", "tree"))
  else if (!is.null(gmodel) && !is.function(gmodel))
    # No further checks here: fingers crossed :)
    stop("gmodel: function(dataHO, targetHO, param) --> function(X) --> y")
  if (is.numeric(params) || is.character(params))
    params <- as.list(params)
  if (!is.list(params) && !is.null(params))
    stop("params: numerical, character, or list (passed to model)")
  if (is.function(gmodel) && !is.list(params))
    stop("params must be provided when using a custom model")
  if (is.list(params) && is.null(gmodel))
    stop("model (or family) must be provided when using custom params")
  if (!is.null(loss) && !is.function(loss))
    # No more checks here as well... TODO:?
    stop("loss: function(y1, y2) --> Real")

  if (is.null(task)) {
    if (is.numeric(target))
      task = "regression"
    else
      task = "classification"
  }
  # Build Model object (= list of parameterized models)
  model <- Model$new(data, target, task, gmodel, params)
  # Return AgghooCV object, to run and predict
  AgghooCV$new(data, target, task, model, loss)
}
