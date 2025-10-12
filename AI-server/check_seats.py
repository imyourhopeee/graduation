from app.models.inference import _engine
import pprint

eng = _engine()
seats = eng.get_seats() or []

print("count:", len(seats))
if seats:
    s0 = seats[0]
    if hasattr(s0, "__dict__"):
        pprint.pprint(vars(s0))
    elif isinstance(s0, dict):
        pprint.pprint(s0)
    else:
        print("첫 좌석:", s0)
