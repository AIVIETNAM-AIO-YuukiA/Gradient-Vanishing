# Follow-up 03: Align learning-curve description and plot

Status: **after behavior-preserving extraction**. Cell 21 describes validation curves as “mean ± std” over seeds, but plots only the mean. The per-seed validation histories already exist.

Choose and implement one explicit presentation contract during the follow-up: display standard-deviation bands (recommended) or describe the figure as mean only. Do not change the training or metric histories. Keep the source notebook unchanged and document the historical figure in the audit.

Regression test: use a small synthetic result fixture with distinct seed curves; assert the reporter computes the sample mean and sample standard deviation at each validation step, and that the plotted uncertainty band (if chosen) uses those values and aligned step positions.
