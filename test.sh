model_dir=ckpt/wmt14_enfr_bpe40k_fp16/8192_8u3
avgs=(`ls ${model_dir}/*avg*.pt`)
echo ${avgs[0]}
echo ${avgs[1]}
if [ ${#avgs[@]$} == 2 ]; then
    exit 0
fi
echo ${#avgs[@]$}
