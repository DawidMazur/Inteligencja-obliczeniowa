import matplotlib.pyplot as plt
import random

from aco import AntColony

plt.style.use("dark_background")

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (50, 10),
    (10, 10),
    (10, 50),
    (10, 90),
    (50, 90),
    (90, 90),
    (90, 50),
    (90, 10),
    (70, 70),
    (70, 30),
    (99, 21),
    (22, 81),
    (34, 19),
    (73, 28),
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,
                   pheromone_evaporation_rate=0.05, pheromone_constant=1000.0,
                   iterations=30)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.show()

# dla m:20, iteracji:50 i rate: 0.20 wynik: 479.22222761204796
# dla m:20, iteracji:50 i rate: 0.40 wynik: 533.6656531812733
# dla m:40, iteracji:50 i rate: 0.40 wynik: 460.397464079333
# dla m:40, iteracji:50 i rate: 0.20 wynik: 488.60517134150393

# dla m:100, iteracji:50 i rate: 0.40 wynik: 452.0139310062922
# dla m:100, iteracji:100 i rate: 0.40 wynik: 521.4907653530026
# wniosek - więcej interacji to niekoniecznie lepszy wynik, przynajmniej nie przy tak mocnym wygasaniu feromonów

# dla m:100, iteracji:100 i rate: 0.20 wynik: 504.8367661256965
# dla m:100, iteracji:100 i rate: 0.05 wynik: 509.80147787412386
# wniosek - przy małym wygasaniu feromonów wynik też może nie być najlepszy bo bardzo utrwala się pierwsza ścieżka

# dla m:30, iteracji:100 i rate: 0.40 wynik: 447.3153869114498
# najlepszy wynik przy mniejszej ilości mrówek póki co, nadal widze jednak że nie idealny


# dla m:30, iteracji:100 i rate: 0.20 wynik: 464.91080545164795
# gorsze wyniki przy wolniejszym wygasaniu feromonów

# dla m:30, iteracji:100 i rate: 0.60 464.91080545164795
# przesadzanie w drugą stronę też psuje wyniki


# dla m:300, iteracji:30 i rate: 0.40 wynik: 460.397464079333
# dla m:300, iteracji:30 i rate: 0.40 wynik: 479.22222761204796
# dla m:300, iteracji:30 i rate: 0.05 wynik: 456.55360846224687
