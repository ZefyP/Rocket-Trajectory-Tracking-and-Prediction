"""Command Line Interface for the trajectory generator"""
import argparse

parser = argparse.ArgumentParser(description='Generate a trajectory.')

parser.add_argument('--verbose,-v', action='store_true' , help='print the flight log during simulation')
parser.add_argument('--output', type=str, help='output file name')
parser.add_argument('--info', action='store_true' , help='show instructions')

args = parser.parse_args()

if not any(vars(args).values()):
    parser.print_help()
else:
    # generate the trajectory based on the given arguments
    pass # replace this with your actual code to generate the trajectory

# Add a usage message
if __name__ == '__main__':
    args = parser.parse_args()

    if args.info:
        print('-----------------------------------------------------------\n')
        print('-----------------------------------------------------------\n')
        print('Usage: trajectory_generator [OPTIONS]\n')
        print('Options:')
        # print('  --step INT       number of time step to compute (in seconds)')
        print('  --output str      output file name')
        print('  -v, --verbose     print the flight log during simulation')

        print('-----------------------------------------------------------\n')
        exit()
    print('[!] Simulation settings:')
    if args.verbose:
        print("Verbose mode is on.")
    
    if args.output:
        print(f"Output file name: {args.output}")
    
    if args.help:
        parser.print_help()

    print('-----------------------------------------------------------\n')
    print('-----------------------------------------------------------\n')
