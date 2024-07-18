import argparse
import os


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu',            type=int,   default=0)

# =================== random seed ================== #
parser.add_argument('--seed',           type=int,   default=1111)

# ==================== dataset ===================== #
parser.add_argument('--train_file',
                    default='./data/bjtraindata.npz')
parser.add_argument('--val_file',
                    default='./data/bjvaldata.npz')
parser.add_argument('--test_file',
                    default='./data/bjtestdata.npz')
parser.add_argument('--nodes',          type=int,   default=9058,
                    help='porto=9795, Beijing=9058')

# parser.add_argument('--train_file',
#                     default='/hdd/home/ll/code/jup/portotraindata.npz')
# parser.add_argument('--val_file',
#                     default='/hdd/home/ll/code/jup/portovaldata.npz')
# parser.add_argument('--test_file',
#                     default='/hdd/home/ll/code/jup/portotestdata.npz')
# parser.add_argument('--nodes',          type=int,   default=9795,
#                     help='porto=9795, Beijing=9058')

# ===================== model ====================== #
parser.add_argument('--latent_dim',     type=int,   default=128)
parser.add_argument('--n_epochs',       type=int,   default=301)
parser.add_argument('--lstm_layers',    type=int,   default=6)
parser.add_argument('--batch_size',     type=int,   default=512)
parser.add_argument('--lr',             type=float, default=0.0004,
                    help='5e-4 for Beijing, 1e-3 for Newyork')
parser.add_argument('--save_epoch_int', type=int,   default=1)
parser.add_argument('--save_folder',                default='saved_models')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
