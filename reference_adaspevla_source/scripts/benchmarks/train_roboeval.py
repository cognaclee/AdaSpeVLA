"""Train on explicitly converted RoboEval measured-pose trajectories."""
import argparse,json
from pathlib import Path
from adaspevla.datasets.hdf5_common import load_hdf5_dataset
from adaspevla.utils.training import train_two_stage,window_options
from scripts.benchmarks.common_args import add_training_arguments

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data_dir',required=True);p.add_argument('--task_name',required=True);add_training_arguments(p,'./checkpoints/roboeval')
    a=p.parse_args();meta=json.loads((Path(a.data_dir)/a.task_name/'manifest.json').read_text())
    options=window_options(a);options['num_views']=3
    ds=load_hdf5_dataset(a.data_dir,a.task_name,14,a.chunk_horizon,a.num_control_points,a.degree,'roboeval',dataset_options=options)
    ds.calibration['source_conversion']=meta
    train_two_stage(a,ds,'roboeval',14)
if __name__=='__main__':main()
