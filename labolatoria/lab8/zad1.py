from simpful import *

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0, 10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0, 0, 15, term="low")
O2 = TriangleFuzzySet(0, 15, 30, term="medium")
O3 = TriangleFuzzySet(15, 30, 30, term="high")
ling_var = LinguisticVariable([O1, O2, O3], universe_of_discourse=[0, 30])
FS.add_linguistic_variable("tip", ling_var)

FS.add_rules([
    "IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
    "IF (service IS average) THEN (tip IS medium)",
    "IF (quality IS good) OR (service IS good) THEN (tip IS high)"
])


# FS.set_variable("quality", 6.5)
# FS.set_variable("service", 9.8)
#
# tip = FS.inference()
#
# print(tip)

def test_tip_value(FS, quality, service):
    FS.set_variable("quality", quality)
    FS.set_variable("service", service)

    return FS.inference()


# Wykres zmiennej lingwistycznej
ling_var.plot()

print(
    test_tip_value(FS, 0, 0),
    test_tip_value(FS, 6.5, 6.5),
    test_tip_value(FS, 6.5, 3.4),
    test_tip_value(FS, 9.8, 9.8),
    test_tip_value(FS, 10, 10),

)
