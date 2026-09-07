"""Ajan cevabını demo için özetler (scripts/demo.sh kullanır).

Ham JSON'da `tool_calls[].result` yüzlerce satır tutuyor; demoda görülmesi
gereken şey cevabın kendisi, hangi araçlarla üretildiği ve denetimden geçip
geçmediği. Ham çıktı için: ./scripts/demo.sh --ham
"""

import json
import sys
import textwrap

d = json.load(sys.stdin)

print(textwrap.fill(d["answer"], 88, initial_indent="  ", subsequent_indent="  "))
print(f"\n  sağlayıcı : {d['provider']}   tur: {d['iterations']}   yedek: {d['fallback']}")
print("  araçlar   : " + " → ".join(c["name"] for c in d["tool_calls"]))
print("  denetim   : " + (", ".join(d["uyarilar"]) if d["uyarilar"] else "uyarı yok"))
