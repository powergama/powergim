import powergim


def test_package_import():
    sip = powergim.SipModel(grid_data=None, parameter_data=None)

    assert isinstance(sip, powergim.SipModel)
