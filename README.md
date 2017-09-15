# NM-Heston

## Description ##
This project implements the Crank-Nicholson solving scheme and four splitting schemes of the Alternating Direction Implicit (ADI) type:
- Douglas
- Craig–Sneyd
- Modified Craig–Sneyd
- Hundsdorfer–Verwer

The program reports each scheme accuracy for a particular example (parameters are located in run.py). It also outputs several graphs:
- price (underlying spot price/variance grid)
- implied volatility
- convergence of the error as the S-axis meshing is refined.

## Requirements ##
python3.6
scipy0.19

## Reference ##
https://wwwf.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/ADI_Heston.pdf
