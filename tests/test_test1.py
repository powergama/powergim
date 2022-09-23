import powergim


def test_package_import():
    sip = powergim.SipModel()

    assert isinstance(sip, powergim.SipModel)
