cd ~/ru_erasmus/data/comp/project/data
DS=tomasslama/indoor-climbing-gym-hold-segmentation

mkdir -p images/bh images/sm images/bh-phone

for n in 0000 0003 0006 0074 0075 0132 0274 0457 0502 0518 0530 0647 0689 0917 1037; do
  kaggle datasets download $DS -f bh/$n.jpg -p images/bh --unzip
done
for n in 082 108; do
  kaggle datasets download $DS -f sm/$n.jpg -p images/sm --unzip
done
for n in 126 182; do
  kaggle datasets download $DS -f bh-phone/$n.jpg -p images/bh-phone --unzip
done

find images -name "*.jpg" | wc -l      # should print 19
