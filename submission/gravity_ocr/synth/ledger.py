"""A made-up site ledger: sectors, drawings, equipment, sensors, pipes, people.

Two domains:

``plant``  a power plant; codes have the KKS shape ``99AAA99AA999`` (unit,
           system, system number, equipment class, serial) and a sensor tag
           shares the first seven characters with its equipment, followed by
           ``C`` and the measured quantity;
``aidc``   an AI data centre; codes have the DCIM shape ``AA9AAA99AA99``
           (site, centre, system, system number, class, serial) and a sensor tag
           shares the first eight characters, followed by ``C`` and the quantity.

Everything else – which systems exist, what they are called, every value and
every person – is drawn from a seed, so two seeds give two sites with no code or
name in common beyond the vocabulary below.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

KKS_MASK = "99AAA99AA999"
KIND_AT = 7  # sensor tags share this many leading characters with their equipment

# quantity → (tag letters, unit, decimals, (low, high) of the normal value, alarm side)
QUANTITIES = {
    "P": ("CP", "bar", 1, (2.0, 180.0), "high"),
    "T": ("CT", "℃", 1, (30.0, 560.0), "high"),
    "F": ("CF", "t/h", 1, (20.0, 1800.0), "low"),
    "L": ("CL", "mm", 0, (300.0, 2400.0), "low"),
    "S": ("CS", "rpm", 0, (600.0, 3600.0), "high"),
    "E": ("CE", "A", 1, (40.0, 2400.0), "high"),
    "V": ("CE", "kV", 1, (3.3, 345.0), "low"),
    "Y": ("CY", "mm/s", 2, (0.4, 4.5), "high"),
}
QUANTITY_LABELS = {
    "P": ["토출 압력", "입구 압력", "출구 압력", "차압", "계통 압력"],
    "T": ["입구 온도", "출구 온도", "베어링 온도", "권선 온도", "유온"],
    "F": ["유량", "토출 유량", "재순환 유량"],
    "L": ["수위", "유면", "탱크 수위"],
    "S": ["회전수", "축 회전수"],
    "E": ["전류", "부하 전류", "여자 전류"],
    "V": ["전압", "모선 전압"],
    "Y": ["축 진동", "베어링 진동"],
}

# data-centre quantities in addition to the plant ones
QUANTITIES.update({
    "G": ("CG", "%", 1, (10.0, 95.0), "high"),
    "W": ("CW", "kW", 1, (20.0, 1800.0), "high"),
})
QUANTITY_LABELS.update({"G": ["밸브 개도", "댐퍼 개도"], "W": ["소비 전력", "랙 전력"]})

# symbol class → (equipment class letters, quantities it may carry)
KINDS = {
    "pump": ("AP", "PFYT"),
    "valve": ("AA", "PF"),
    "turbine": ("AD", "STYP"),
    "fan": ("AN", "SYFE"),
    "exchanger": ("AC", "TP"),
    "vessel": ("BB", "LPT"),
    "breaker": ("GS", "EV"),
    "transformer": ("GT", "TVE"),
    "generator": ("GA", "EVTY"),
    "panel": ("GH", "VE"),
}

# sector key → building, label, medium, systems (code, label), equipment (kind, name)
SECTORS = {
    "feed": ("터빈동", "급수", "water", [("LAB", "급수"), ("LAC", "급수펌프"), ("LAD", "고압 급수가열")], [
        ("vessel", "탈기기 저장탱크"), ("pump", "보일러 급수펌프"), ("valve", "급수 차단밸브"),
        ("exchanger", "고압 급수가열기"), ("valve", "급수 조절밸브"), ("pump", "급수 승압펌프"),
        ("valve", "재순환 밸브"), ("panel", "급수 제어반"), ("exchanger", "급수 냉각기"), ("vessel", "급수 완충탱크"),
    ]),
    "steam": ("보일러동", "주증기", "steam", [("LBA", "주증기"), ("HAH", "과열기"), ("LBC", "재열증기")], [
        ("vessel", "기수분리기"), ("exchanger", "1차 과열기"), ("exchanger", "최종 과열기"),
        ("valve", "주증기 차단밸브"), ("valve", "주증기 조절밸브"), ("exchanger", "재열기"),
        ("valve", "안전밸브"), ("vessel", "드레인 탱크"), ("panel", "증기 제어반"), ("valve", "우회 밸브"),
    ]),
    "turbine": ("터빈동", "터빈", "shaft", [("MAA", "고압 터빈"), ("MAC", "저압 터빈"), ("MAG", "복수기")], [
        ("valve", "터빈 정지밸브"), ("turbine", "고압 터빈"), ("turbine", "중압 터빈"),
        ("turbine", "저압 터빈"), ("exchanger", "복수기"), ("pump", "진공 펌프"),
        ("panel", "터빈 감시반"), ("valve", "추기 밸브"), ("vessel", "배수 탱크"), ("fan", "글랜드 배풍기"),
    ]),
    "cooling": ("취수동", "순환수", "water", [("PAC", "순환수 펌프"), ("PAB", "순환수 배관"), ("PCB", "기기 냉각수")], [
        ("vessel", "취수 수조"), ("pump", "순환수 펌프"), ("valve", "순환수 토출밸브"),
        ("exchanger", "기기 냉각수 열교환기"), ("pump", "기기 냉각수 펌프"), ("fan", "냉각탑 송풍기"),
        ("valve", "방류 밸브"), ("panel", "취수 제어반"), ("vessel", "냉각수 팽창탱크"), ("valve", "역세 밸브"),
    ]),
    "draft": ("보일러동", "통풍", "draft", [("HLB", "압입 송풍"), ("HNC", "유인 송풍"), ("HFC", "미분기")], [
        ("fan", "압입 송풍기"), ("exchanger", "공기 예열기"), ("fan", "1차 공기 송풍기"),
        ("vessel", "미분기"), ("valve", "풍량 조절댐퍼"), ("fan", "유인 송풍기"),
        ("exchanger", "가스 냉각기"), ("panel", "연소 제어반"), ("valve", "차단 댐퍼"), ("vessel", "집진기 호퍼"),
    ]),
    "power": ("전기동", "전력", "power", [("MKA", "발전기"), ("BAT", "주변압기"), ("BBA", "고압 배전")], [
        ("generator", "주 발전기"), ("breaker", "발전기 차단기"), ("transformer", "주변압기"),
        ("breaker", "송전 차단기"), ("transformer", "소내 변압기"), ("panel", "고압 배전반"),
        ("breaker", "모선 연락 차단기"), ("panel", "보호 계전반"), ("transformer", "여자 변압기"), ("generator", "비상 발전기"),
    ]),
    "lube": ("터빈동", "윤활유", "oil", [("MAV", "윤활유"), ("MAX", "제어유")], [
        ("vessel", "윤활유 탱크"), ("pump", "주 윤활유 펌프"), ("exchanger", "윤활유 냉각기"),
        ("valve", "유압 조절밸브"), ("pump", "비상 윤활유 펌프"), ("vessel", "제어유 축압기"),
        ("fan", "유증기 배풍기"), ("panel", "윤활유 감시반"), ("valve", "유온 조절밸브"), ("pump", "제어유 펌프"),
    ]),
    "cond": ("터빈동", "복수", "water", [("LCA", "복수"), ("LCB", "복수펌프"), ("LCC", "저압 급수가열")], [
        ("vessel", "복수 저장조"), ("pump", "복수 펌프"), ("valve", "복수 차단밸브"),
        ("exchanger", "저압 급수가열기"), ("exchanger", "글랜드 증기 복수기"), ("pump", "복수 승압펌프"),
        ("valve", "수위 조절밸브"), ("panel", "복수 제어반"), ("vessel", "보충수 탱크"), ("valve", "배수 밸브"),
    ]),
}  # fmt: skip
# AI data centre: sector key → building, label, medium, systems, equipment
AIDC_SECTORS = {
    "compute": ("101동", "GPU 컴퓨트", "coolant", [("GPU", "GPU 랙"), ("MNF", "인랙 매니폴드"), ("RTN", "환수 매니폴드")], [
        ("rack", "GPU 랙 Pod"), ("rack", "추론 랙 Pod"), ("manifold", "인랙 매니폴드"), ("manifold", "환수 매니폴드"),
        ("valve", "매니폴드 차단밸브"), ("valve", "유량 조절밸브"), ("panel", "랙 감시반"), ("fan", "후면 도어 팬"),
        ("vessel", "냉각수 버퍼탱크"), ("pump", "랙 순환펌프"),
    ]),
    "cooling": ("103동", "냉각수 분배", "coolant", [("CDU", "냉각수 분배장치"), ("CHW", "냉수"), ("CHL", "칠러")], [
        ("exchanger", "CDU 열교환기"), ("pump", "CDU 1차 펌프"), ("pump", "CDU 2차 펌프"), ("valve", "CDU 2차측 유량조절밸브"),
        ("vessel", "CDU 팽창탱크"), ("exchanger", "칠러 증발기"), ("pump", "냉수 펌프"), ("valve", "냉수 바이패스밸브"),
        ("panel", "CDU 제어반"), ("vessel", "냉수 저장조"),
    ]),
    "tower": ("105동", "냉각탑", "water", [("CTW", "냉각탑"), ("CHL", "칠러"), ("CWS", "냉각수 공급")], [
        ("fan", "냉각탑 송풍기"), ("pump", "냉각수 펌프"), ("exchanger", "칠러 응축기"), ("valve", "냉각수 차단밸브"),
        ("vessel", "냉각탑 수조"), ("panel", "냉각탑 제어반"), ("valve", "보충수 밸브"), ("pump", "보충수 펌프"),
        ("exchanger", "판형 열교환기"), ("fan", "환기 팬"),
    ]),
    "power": ("104동", "전력", "power", [("SWG", "수배전"), ("UPS", "무정전전원"), ("PDU", "전력 분배")], [
        ("transformer", "수전 변압기"), ("breaker", "수전 차단기"), ("ups", "UPS 모듈"), ("battery", "배터리 스트링"),
        ("breaker", "PDU 차단기"), ("panel", "PDU 분전반"), ("transformer", "몰드 변압기"), ("panel", "보호 계전반"),
        ("ups", "정류기 모듈"), ("breaker", "모선 연락 차단기"),
    ]),
    "facility": ("106동", "공조", "air", [("CRA", "항온항습기"), ("IRC", "인로우 쿨러"), ("AHU", "공조기")], [
        ("fan", "항온항습기 팬"), ("fan", "인로우 쿨러 팬"), ("exchanger", "공조기 코일"), ("valve", "냉수 조절밸브"),
        ("panel", "공조 제어반"), ("fan", "급기 팬"), ("fan", "배기 팬"), ("vessel", "가습기 수조"),
        ("valve", "댐퍼"), ("exchanger", "재열 코일"),
    ]),
    "network": ("102동", "네트워크", "data", [("FAB", "패브릭"), ("OPT", "광전송")], [
        ("switch", "스파인 스위치"), ("switch", "리프 스위치"), ("switch", "광 전송장치"), ("panel", "광분배반"),
        ("rack", "네트워크 랙"), ("fan", "랙 배기 팬"), ("panel", "네트워크 감시반"), ("switch", "관리 스위치"),
    ]),
}  # fmt: skip
AIDC_KINDS = {
    "rack": ("RK", "TWE"), "manifold": ("MV", "PTF"), "pump": ("AP", "PFE"), "valve": ("AA", "GF"),
    "exchanger": ("AC", "TP"), "vessel": ("BB", "LP"), "fan": ("AN", "SE"), "panel": ("GH", "VE"),
    "breaker": ("GS", "EV"), "transformer": ("GT", "TVE"), "ups": ("GU", "WVE"), "battery": ("BT", "VT"),
    "switch": ("SW", "TW"),
}  # fmt: skip

MEDIUM_LABELS = {
    "water": ["급수", "보충수", "냉각수", "순환수"],
    "steam": ["주증기", "추기", "재열증기"],
    "shaft": ["축 동력", "배기"],
    "draft": ["연소공기", "연소가스"],
    "power": ["전력", "제어 전원"],
    "oil": ["윤활유", "제어유"],
    "coolant": ["2차 냉각수", "공급 냉각수", "환수"],
    "air": ["급기", "환기"],
    "data": ["광 회선", "데이터 링크"],
}
FLOORS = [("B1", "지하 1층", -4.5), ("1F", "1층", 0.0), ("2F", "2층", 8.5), ("3F", "3층", 17.0), ("RF", "옥상", 25.5)]
SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민"
GIVEN = [
    "민준", "서연", "도윤", "지우", "하준", "서현", "시우", "지민", "주원", "수아", "지호", "하은", "예준", "윤서",
    "건우", "채원", "현우", "지유", "우진", "소율", "선우", "다은", "연우", "예린", "정훈", "미경", "상철", "영숙",
    "동현", "은정", "재석", "혜진", "성민", "보람", "태윤", "가은", "경수", "나연", "병호", "유진",
]  # fmt: skip
ROLES = ["기술책임자", "운전원", "정비원", "계측 담당", "안전 관리자"]
SUFFIXES = ["A", "B", "C", "1호", "2호", ""]


@dataclass
class Sensor:
    tag: str
    kks: str
    quantity: str
    label: str
    unit: str
    decimals: int
    normal: float
    alarm: float
    trip: float
    side: str  # which side of normal the limits are on: "high" or "low"


@dataclass
class Equipment:
    kks: str
    name: str
    sector: str
    floor: str
    drawing: str
    kind: str
    cell: tuple  # (column, row) on the drawing grid


@dataclass
class Edge:
    src: str
    dst: str
    medium: str
    label: str


@dataclass
class Sector:
    key: str
    code: str
    building: str
    label: str
    drawing: str


@dataclass
class Drawing:
    id: str
    sector: str
    title: str
    rev: str
    issued: str
    signer: str
    layout: str  # "top": title bar along the top; "corner": title block bottom right
    jump_in: str = ""
    jump_out: str = ""


@dataclass
class Person:
    id: str
    name: str
    role: str


@dataclass
class Moc:
    moc_no: str
    kks: str
    tag: str
    param: str
    old: float
    new: float
    unit: str
    approved_by: str
    approved_on: str


@dataclass
class Ledger:
    seed: int
    domain: str = "plant"  # "plant" (KKS codes) or "aidc" (DCIM codes)
    sectors: list[Sector] = field(default_factory=list)
    drawings: list[Drawing] = field(default_factory=list)
    equipment: list[Equipment] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    people: list[Person] = field(default_factory=list)
    mocs: list[Moc] = field(default_factory=list)

    @property
    def scheme(self):
        from ..drawing.codes import DCIM, KKS

        return DCIM if self.domain == "aidc" else KKS

    @property
    def code_label(self) -> str:
        return self.scheme.label

    # ------------------------------------------------------------------ lookups
    def equipment_of(self, drawing: str) -> list[Equipment]:
        return [e for e in self.equipment if e.drawing == drawing]

    def sensors_of(self, kks: str) -> list[Sensor]:
        return [s for s in self.sensors if s.kks == kks]

    def edges_of(self, drawing: str) -> list[Edge]:
        codes = {e.kks for e in self.equipment_of(drawing)}
        return [e for e in self.edges if e.src in codes and e.dst in codes]

    def drawing(self, drawing_id: str) -> Drawing:
        return next(d for d in self.drawings if d.id == drawing_id)

    def sector(self, key: str) -> Sector:
        return next(s for s in self.sectors if s.key == key)

    def person(self, person_id: str) -> Person:
        return next(p for p in self.people if p.id == person_id)

    def location(self, e: Equipment) -> str:
        code, label, el = next(f for f in FLOORS if f[0] == e.floor)
        return f"{self.sector(e.sector).building} {label} (EL.{el:+.1f}m)"

    @property
    def codes(self) -> list[str]:
        return [e.kks for e in self.equipment]

    @property
    def tags(self) -> list[str]:
        return [s.tag for s in self.sensors]

    def to_json(self) -> dict:
        return asdict(self)


def fmt_value(v: float, decimals: int | None = None) -> str:
    """How a reading is printed: thousands separators from 1,000 up."""
    if decimals is not None:
        return f"{v:,.{decimals}f}"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    if abs(v) >= 1 or v == 0:
        return f"{v:.1f}"
    return f"{v:.2f}"


def short_tag(tag: str) -> str:
    """The part of a sensor tag from its quantity letters on (``CP003``, ``CG02``)."""
    from ..drawing.codes import scheme_of

    sc = scheme_of(tag)
    return tag[sc.kind_at if sc else KIND_AT :]


def snake(n: int, cols: int = 4, rows: int = 3) -> list[tuple]:
    """Grid cells in boustrophedon order, so that consecutive equipment are neighbours."""
    cells = []
    for r in range(rows):
        order = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        cells += [(c, r) for c in order]
    return cells[:n]


def make_ledger(seed: int, sectors: int = 4, layouts: tuple = ("top", "corner"), domain: str = "plant") -> Ledger:
    rng = random.Random(seed)
    led = Ledger(seed, domain)
    aidc = domain == "aidc"
    table = AIDC_SECTORS if aidc else SECTORS
    kinds = AIDC_KINDS if aidc else KINDS
    keys = rng.sample(sorted(table), sectors)
    unit = rng.choice(["DC1", "DC2"]) if aidc else rng.choice([10, 20, 30, 40, 50, 60, 70, 80])
    kind_at = 8 if aidc else KIND_AT

    names = set()
    while len(names) < 8:
        names.add(rng.choice(SURNAMES) + rng.choice(GIVEN))
    for i, name in enumerate(sorted(names)):
        role = ROLES[0] if i < 2 else rng.choice(ROLES[1:])
        led.people.append(Person(f"E{rng.randint(1000, 9999)}", name, role))
    chiefs = [p for p in led.people if p.role == ROLES[0]]

    base_no = rng.randint(1, 8) * 100
    for si, key in enumerate(keys):
        building, label, medium, systems, templates = table[key]
        prefix = "MEP" if aidc else "PID"
        drawing_id = f"{prefix}-{base_no + si * 20 + rng.randint(1, 15):03d}"
        led.sectors.append(Sector(key, f"{unit}{systems[0][0]}", building, label, drawing_id))
        issued = f"20{rng.randint(22, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        title = f"{label} 계통도" if aidc else f"{label} 계통 배관 계장도"
        led.drawings.append(
            Drawing(
                drawing_id, key, title, rng.choice("ABCDEFG"), issued,
                rng.choice(chiefs).id, layouts[si % len(layouts)],
            )  # fmt: skip
        )
        count = rng.randint(6, 9)
        picks = rng.sample(templates, count)
        cells = sorted(rng.sample(range(12), count))
        order = snake(12)
        serial: dict = {}
        system_no = {code: rng.choice([10, 20, 30] if aidc else [10, 11, 12, 20, 21, 30, 40, 50]) for code, _ in systems}
        previous = None
        for (kind, stem), ci in zip(picks, cells):
            system = rng.choice(systems)[0]
            letters, quantities = kinds[kind]
            n = serial.get((system, letters), rng.randint(1, 3))
            serial[(system, letters)] = n + rng.randint(1, 4 if not aidc else 2)
            serial_txt = f"{n % 100:02d}" if aidc else f"{n:03d}"
            kks = f"{unit}{system}{system_no[system]:02d}{letters}{serial_txt}"
            suffix = rng.choice(SUFFIXES) if kind in ("pump", "fan", "valve", "breaker") else ""
            name = f"{stem} {suffix}".strip()
            eq = Equipment(kks, name, key, rng.choice(FLOORS)[0], drawing_id, kind, order[ci])
            led.equipment.append(eq)
            used_tags = set()
            for q in rng.sample(quantities, min(len(quantities), rng.randint(1, 2))):
                letters2, unit_text, decimals, (lo, hi), side = QUANTITIES[q]
                tag = f"{kks[:kind_at]}{letters2}{rng.randint(1, 9):02d}" if aidc else f"{kks[:KIND_AT]}{letters2}{rng.randint(1, 30):03d}"
                if tag in used_tags or tag in led.tags:
                    continue
                used_tags.add(tag)
                normal = round(rng.uniform(lo, hi), decimals)
                step = 1.0 + rng.uniform(0.06, 0.18)
                if side == "high":
                    alarm, trip = normal * step, normal * (step + rng.uniform(0.05, 0.15))
                else:
                    alarm, trip = normal / step, normal / (step + rng.uniform(0.05, 0.15))
                led.sensors.append(
                    Sensor(
                        tag, kks, q, rng.choice(QUANTITY_LABELS[q]), unit_text, decimals,
                        normal, round(alarm, decimals), round(trip, decimals), side,
                    )  # fmt: skip
                )
            if previous is not None:
                led.edges.append(Edge(previous.kks, kks, medium, rng.choice(MEDIUM_LABELS[medium])))
            previous = eq

    for i, d in enumerate(led.drawings):
        d.jump_in = led.drawings[i - 1].id if i else ""
        d.jump_out = led.drawings[i + 1].id if i + 1 < len(led.drawings) else ""

    year = rng.randint(2024, 2026)
    for i, d in enumerate(led.drawings):
        with_sensors = [e for e in led.equipment_of(d.id) if led.sensors_of(e.kks)]
        if not with_sensors:
            continue
        eq = rng.choice(with_sensors)
        s = rng.choice(led.sensors_of(eq.kks))
        new = round(s.alarm * rng.choice([0.9, 0.94, 1.05, 1.1]), s.decimals)
        if new == s.alarm:
            new = round(s.alarm + 10 ** -s.decimals, s.decimals)
        led.mocs.append(
            Moc(
                f"MOC-{year}-{rng.randint(1, 99):03d}", eq.kks, s.tag, f"{s.label} 경보", s.alarm, new, s.unit,
                rng.choice(chiefs).id, f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            )  # fmt: skip
        )
    return led
