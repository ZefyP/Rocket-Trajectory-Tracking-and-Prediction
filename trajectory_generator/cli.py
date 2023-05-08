"""Command Line Interface for the trajectory generator"""
import argparse

parser = argparse.ArgumentParser(description='Generate a trajectory.')

parser.add_argument('--verbose,-v', type=str, help='print the flight log during simulation')
parser.add_argument('--output', type=str, help='output file name')
parser.add_argument('--help', type=str, help='show instructions')

# parser.add_argument('--velocity', type=float, help='velocity of the object')
# parser.add_argument('--angle', type=float, help='angle of projection')
# parser.add_argument('--height', type=float, help='height of projection')
# parser.add_argument('--time', type=float, help='total time of flight')
# parser.add_argument('--steps', type=int, help='number of time steps to compute')

args = parser.parse_args()

if args.velocity is None or args.angle is None or args.height is None or args.time is None or args.steps is None or args.output is None:
    parser.print_help()
else:
    # generate the trajectory based on the given arguments
    pass # replace this with your actual code to generate the trajectory
