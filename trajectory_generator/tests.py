from trajectory_generator.stage import Stage
from trajectory_generator.parachute import Parachute

# Create two parachute objects
para1 = Parachute("Drogue", 0.75, "circle", lambda p, s: p < 101325)
para2 = Parachute("Main", 4.0, "ellipse", lambda p, s: p < 101325 and s[1] < 1000)

# Create a stage object and add the parachutes to it
stage1 = Stage("First Stage", 10000, 100000, 1000000, 10.0,
                diameter=5.0, length=20.0, parachutes=[para1, para2])

# Print out the stage attributes
print(f"Stage name: {stage1.name}")
print(f"Stage dry mass: {stage1.dry_mass}")
print(f"Stage fuel mass: {stage1.fuel_mass}")
print(f"Stage total mass: {stage1.total_mass}")
print(f"Stage thrust: {stage1.thrust}")
print(f"Stage burn time: {stage1.burn_time}")
print(f"Stage separation time: {stage1.separation_time}")
print(f"Stage diameter: {stage1.diameter}")
print(f"Stage length: {stage1.length}")
print(f"Stage cross-sectional area: {stage1.cross_sectional_area}")
print(f"Stage KML colour: {stage1.kml_colour}")

# Print out the parachute attributes
for chute in stage1.parachutes:
    print(f"\nParachute name: {chute.name}")
    print(f"Parachute diameter: {chute.diameter}")
    print(f"Parachute shape: {chute.shape}")
    print(f"Parachute lag time: {chute.lag}")



# from trajectory_generator.altitude import Altitude


# def test_is_at_terminal_velocity():
#     """ Testing the terminal velocity function"""
    
#     model = Altitude(DESCENT_MODE_NORMAL, 3000.0, 10.0, 0.75)
#     # assuming initial altitude is 10000 meters and initial vertical velocity is 0
#     v_z = -model.drag_coeff / math.sqrt(get_density(10000))
#     assert model.is_at_terminal_velocity(V_z) == False
#     # after some time, the vertical velocity should approach terminal velocity
#     V_z = -model.drag_coeff / math.sqrt(get_density(7000))
#     assert model.is_at_terminal_velocity(V_z) == True


# test_is_at_terminal_velocity()