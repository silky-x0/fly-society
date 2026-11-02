# Reproduce Shiu et al. example.ipynb sugarR experiment (v630 data).
# Run from the Shiu repo: cd ~/Downloads/Drosophila_brain_model && PYTHONPATH=. python <this file> <n_run> <n_proc> <exp_name>
import sys, time, resource
from model import run_exp, default_params as params

neu_sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570]
params['n_run'] = int(sys.argv[1])
t = time.time()
run_exp(exp_name=sys.argv[3], neu_exc=neu_sugar, params=params, n_proc=int(sys.argv[2]),
        path_res='./results/example',
        path_comp='./2023_03_23_completeness_630_final.csv',
        path_con='./2023_03_23_connectivity_630_final.parquet', force_overwrite=True)
print(f'wall {time.time()-t:.0f}s  peak RSS self {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3:.0f} MB  children {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e3:.0f} MB')
