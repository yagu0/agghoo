#' agghoo
#'
#' Run the (core) agghoo procedure.
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
#' @seealso Function \code{\link{compareTo}}
#'
#' @references
#' Guillaume Maillard, Sylvain Arlot, Matthieu Lerasle. "Aggregated hold-out".
#' Journal of Machine Learning Research 22(20):1--55, 2021.
#'
#' @export
agghoo <- function(
  data, target, task = NULL, gmodel = NULL, params = NULL, loss = NULL
) {
	# Args check:
  checkDaTa(data, target)
  task <- checkTask(task, target)
  modPar <- checkModPar(gmodel, params)
  loss <- checkLoss(loss, task)

  # Build Model object (= list of parameterized models)
  model <- Model$new(data, target, task, modPar$gmodel, modPar$params)

  # Return AgghooCV object, to run and predict
  AgghooCV$new(data, target, task, model, loss)
}
