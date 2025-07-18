



# python bsdf_optimization.py --scene tray --iter 100 --spp 2 --use_prb --loss RelativeL2 --lr 0.02
# python bsdf_optimization.py --scene tray --iter 100 --spp 2 --use_biased --use_cb --loss RelativeL2 --lr 0.02
# python bsdf_optimization.py --scene tray --iter 100 --spp 2 --use_biased --use_cb --use_js --loss RelativeL2 --lr 0.02

# python bsdf_optimization.py --scene plane --iter 100 --spp 8 --use_prb --loss RelativeL2 --lr 0.02
# python bsdf_optimization.py --scene plane --iter 100 --spp 8 --use_biased --use_cb --loss RelativeL2 --lr 0.02
# python bsdf_optimization.py --scene plane --iter 100 --spp 8 --use_biased --use_cb --use_js --loss RelativeL2 --lr 0.02

python bsdf_optimization.py --scene frame --iter 100 --spp 8 --use_prb --loss RelativeL2 --lr 0.02
python bsdf_optimization.py --scene frame --iter 100 --spp 8 --use_biased --use_cb --loss RelativeL2 --lr 0.02
python bsdf_optimization.py --scene frame --iter 100 --spp 8 --use_biased --use_cb --use_js --loss RelativeL2 --lr 0.02

