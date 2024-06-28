# Load necessary libraries
library(rms)
library(survival)
library(dplyr)
library(caret)
library(ggplot2)
library(officer)
library(flextable)
library(regplot)
library(pROC)
library(rmda)

# Set working directory
setwd("your path here")

# Load data
mydata <- read.csv("your csv here", header = TRUE)

# Get variable names
predictors <- colnames(mydata)[!colnames(mydata) %in% "Group"]

# Univariate logistic regression analysis
univariate_results <- list()
significant_vars <- c()
for (predictor in predictors) {
  formula <- as.formula(paste("Group ~", predictor))
  model <- glm(formula, data = mydata, family = binomial)
  summary_model <- summary(model)
  univariate_results[[predictor]] <- summary_model
  if (summary_model$coefficients[2, 4] < 0.05) {  # Check if p-value is less than 0.05
    significant_vars <- c(significant_vars, predictor)
  }
}

# Display univariate regression results
for (predictor in predictors) {
  cat("\nUnivariate regression results:", predictor, "\n")
  print(univariate_results[[predictor]])
}

# Create Word document
doc <- read_docx()

# Add univariate analysis results to Word document
doc <- doc %>%
  body_add_par("Univariate Regression Analysis Results", style = "heading 1")

for (predictor in predictors) {
  summary_model <- univariate_results[[predictor]]
  OR <- exp(coef(summary_model)[2, 1])
  p_value <- coef(summary_model)[2, 4]
  result_df <- data.frame(
    Variable = predictor,
    OR = OR,
    P.value = p_value
  )
  ft <- flextable(result_df)
  doc <- doc %>%
    body_add_par(paste("Variable:", predictor), style = "heading 2") %>%
    body_add_flextable(ft)
}

# Check significant variables
cat("Significant variables:", significant_vars, "\n")

# Multivariate logistic regression analysis
if (length(significant_vars) > 0) {
  formula_str <- paste("Group ~", paste(significant_vars, collapse = " + "))
  cat("Multivariate regression formula:", formula_str, "\n")
  fml <- as.formula(formula_str)
  print(fml)
  
  multifit <- glm(fml, data = mydata, family = binomial)
  
  # Get multivariate regression results
  coefficients <- summary(multifit)$coefficients
  
  # Create multivariate regression summary table
  multi_factor_df <- data.frame(
    Variable = rownames(coefficients),
    OR = exp(coefficients[, 1]),
    P.value = coefficients[, 4]
  )
  ft_multi <- flextable(multi_factor_df)
  doc <- doc %>%
    body_add_par("Multivariate Regression Analysis Results", style = "heading 1") %>%
    body_add_flextable(ft_multi)
  
  # Save Word document
  print(doc, target = "Regression_Results.docx")
  
  # Select significant variables from multivariate regression
  final_vars <- rownames(coefficients)[coefficients[, 4] < 0.05]
  cat("Significant variables in multivariate regression:", final_vars, "\n")
  
  if (length(final_vars) > 0) {
    final_formula_str <- paste("Group ~", paste(final_vars, collapse = " + "))
    final_fml <- as.formula(final_formula_str)
    print(final_fml)
    
    dd <- datadist(mydata)
    options(datadist = "dd")
    
    final_model <- lrm(final_fml, data = mydata, x = TRUE, y = TRUE)
    print(final_model)
    
  # Create interactive nomogram
    regplot(final_model, 
            observation = mydata[1,],
            plots = c("density", "boxes"),
            center = TRUE,
            title = "Nomogram",
            points = TRUE,
            odds = FALSE,
            showP = FALSE,
            droplines = FALSE,
            rank = "sd",
            clickable = FALSE,
            dencol = "#00bdcd", boxcol = "#f88421")
    
# Plot ROC curve
    prob <- predict(final_model, mydata, type = "fitted")
    roc_curve <- roc(mydata$Group, prob)
    pdf("ROC_Curve.pdf", width = 12, height = 8)
    plot.roc(roc_curve,
             print.auc = TRUE,
             auc.polygon = TRUE,
             max.auc.polygon = TRUE,
             auc.polygon.col = "skyblue",
             print.thres = TRUE,
             main = "ROC Curve",
             col = "red",
             print.thres.col = "black",
             identity.col = "blue",
             identity.lty = 1,
             identity.lwd = 1)
    dev.off()
    
    # Calculate AUC and 95% CI
    roc_curve$auc
    ci(roc_curve)
    # Get optimal cutoff value
    cutoff <- coords(roc_curve, "best", ret = "threshold")
    cutoff
    # Get specificity at optimal cutoff value
    spe <- coords(roc_curve, "best")$specificity
    spe
    # Get sensitivity at optimal cutoff value
    sen <- coords(roc_curve, "best")$sensitivity
    sen
    
    # Calibration curve
    cal <- calibrate(final_model, method = 'boot', m = 50, B = 1000)
    plot(cal,
         xlim = c(0, 1),
         xlab = "Predicted Probability",
         ylab = "Observed Probability",
         legend = FALSE,
         subtitles = FALSE)
    abline(0, 1, col = "black", lty = 2, lwd = 2)
    lines(cal[,c("predy","calibrated.orig")], 
          type = "l", lwd = 2, col = "red", pch = 16)
    lines(cal[,c("predy","calibrated.corrected")], 
          type = "l", lwd = 2, col = "green", pch = 16)
    legend(0.55, 0.35,
           c("Apparent", "Ideal", "Bias-corrected"),
           lty = c(2, 1, 1),
           lwd = c(2, 1, 1),
           col = c("black", "red", "green"),
           bty = "n") 
    
    # Decision Curve Analysis (DCA)
    dca_result <- decision_curve(Group ~  #your own formula here,
                                 data = mydata, family = binomial(link = "logit"),
                                 thresholds = seq(0, 1, by = 0.01),
                                 confidence.intervals = 0.95,
                                 study.design = "case-control",
                                 population.prevalence = 0.3)
    plot_decision_curve(dca_result,
                        curve.names = "Model",
                        cost.benefit.axis = F,
                        col = "red",
                        confidence.intervals = F,
                        standardize = T,
                        legend.position = "topright")
  }
}