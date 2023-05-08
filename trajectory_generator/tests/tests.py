from trajectory_generator.altitude import Altitude


def test_is_at_terminal_velocity():
    """ Testing the terminal velocity function"""
    
    model = Altitude(DESCENT_MODE_NORMAL, 3000.0, 10.0, 0.75)
    # assuming initial altitude is 10000 meters and initial vertical velocity is 0
    v_z = -model.drag_coeff / math.sqrt(get_density(10000))
    assert model.is_at_terminal_velocity(V_z) == False
    # after some time, the vertical velocity should approach terminal velocity
    V_z = -model.drag_coeff / math.sqrt(get_density(7000))
    assert model.is_at_terminal_velocity(V_z) == True


test_is_at_terminal_velocity()