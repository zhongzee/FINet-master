# FINET: HARMONIZING FOR DEFECT VISIBILITY WITH FOCUSED INTERACTION
NETWORK

\begin{aligned}
&\text { Table 1: Embedding FINet into Existing Architectures }\\
&\begin{array}{|c|c|c|c|c|c|c|}
\hline \text { methods } & \text { |backbone| } & + FIN & \operatorname{m} / AP50 & mAP & \text { nAP50:5: } & \text { Parameters } \\
\hline \text { TOOD }  & \operatorname{Res} 50 & x & \text { 78.000\% } & 60.40 \% & 55.50 \% & 32.37 M \\
\hline \text { TOOD }  & \operatorname{Res} 50 & \checkmark & 83.10 \% & 62.40 \% & 57.40 \% & 32.61 M \\
\hline \text { DeformDETR }  & \operatorname{Res} 50 & x & 76.7065 & 54.80 \% & 52.30 \% & 32.08 M \\
\hline \text { DeformDETR } & \operatorname{Res} 50 & \checkmark & 82.90 \% & 57.20 \% & 53.60 \% & 32.32 M \\
\hline \text { TOOD } & \text { Swin-S } & x & 79.90 \% & 62.20 \% & 56.70 \% & 57.4 M \\
\hline \text { TOOD } & \text { Swin-S } & \checkmark & 90.50 \% & 65.30 \% & 62.80 \% & 57.64 M \\
\hline
\end{array}
\end{aligned}


\begin{aligned}
&\text { Table 2: Analysis of FFN structures }\\
&\begin{array}{l|l|l|l|l}
\hline \text { FFN } & \text { mAP50 } & \text { mAP75 } & \text { mAP50:5:95 } & \text { Parameters } \\
\hline[ M , M , M , M , M ] & 74.40 \% & 53.80 \% & 51.00 \% & 0.454 M \\
{[ F , F , F , F , F ]} & 81.80 \% & 6 5 . 6 0 \% & 58.80 \% & 0.054 M \\
{[ F , M , M , M , M ]} & 90.00 \% & 64.00 \% & 60.40 \% & 0.374 M \\
{[ F , F , M , M , M ]} & 9 0 . 5 0 \% & 65.30 \% & 6 2 . 8 0 \% & 0.294 M \\
{[ F , F , F , M , M ]} & 83.70 \% & 62.10 \% & 58.60 \% & 0.214 M \\
{[ F , F , F , F , M ]} & 82.22 \% & 63.20 \% & 58.00 \% & 0.134 M \\
\hline
\end{array}
\end{aligned}
