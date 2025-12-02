import math
import random
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# Configurare
NUM_CLIENTS = 40  # Numar clienti (fara depozit; nodul 0 este depozitul)
VEHICLE_CAPACITY = 20  # Q (Capacitatea maxima de incarcare a fiecarei masini)
MAX_VEHICLES = 20  # K (Numarul maxim de masini disponibile in flota)
GENERATIONS = 200  # Numar generatii (cat timp ruleaza algoritmul)
POPULATION_SIZE = 50  # mu (Marimea populatiei de solutii parinti)
OFFSPRING_SIZE = 40  # lambda (Numarul de solutii copii generate per generatie)
ELITISM_COUNT = 1  # E (Numarul celor mai buni indivizi care trec automat in generatia urmatoare)

# Coeficienti penalizare
ALPHA = 1000.0  # Penalizare depasire capacitate (valoare mare pentru a forta respectarea limitei Q)
BETA = 1000.0  # Penalizare vehicule in exces (valoare mare pentru a nu folosi mai mult de K masini)


# Structuri de date
@dataclass
class Node:
    """ Reprezinta un punct pe harta (depozit sau client) """
    id: int
    x: float
    y: float
    demand: int  # Cererea clientului (cat trebuie livrat)


@dataclass
class Individual:
    """ Reprezinta o solutie posibila in algoritmul genetic """
    chromosome: List[int]  # Permutare a ID-urilor clientilor (ordinea de vizitare)
    fitness: float = 0.0  # Calitatea solutiei (cu cat mai mare, cu atat mai bine)
    total_cost: float = 0.0  # Cost real (distanta totala parcursa)
    penalized_cost: float = 0.0  # Costul ajustat cu penalizari (folosit pentru calculul fitness-ului)
    routes: List[List[Node]] = None  # Listele de rute decodificate efectiv


# Generare date
def generate_problem(num_clients, max_demand):
    """ Genereaza coordonate si cereri aleatorii pentru testare """
    depot = Node(0, 50, 50, 0)  # Depozitul este plasat fix in centrul hartii (50, 50)
    clients = []
    for i in range(1, num_clients + 1):
        clients.append(Node(
            i,
            random.uniform(0, 100),  # Coordonata X aleatorie
            random.uniform(0, 100),  # Coordonata Y aleatorie
            random.randint(1, max_demand)  # Cerere aleatorie intre 1 si max_demand
        ))
    return depot, clients


def euclidean_distance(n1: Node, n2: Node):
    """ Calculeaza distanta in linie dreapta intre doua puncte """
    return math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)


# Vizualizare
def plot_routes(depot, routes):
    """ Deseneaza harta finala cu rutele """
    colors = ['blue', 'green', 'red', 'cyan', 'purple', 'yellow', 'black', 'orange', 'gray', 'pink']
    plt.figure(figsize=(10, 8))

    # 1. Desenam Depozitul
    plt.plot(depot.x, depot.y, 'rs', markersize=15, label='Depozit')  # 'rs' = red square

    # 2. Desenam fiecare ruta
    for idx, route in enumerate(routes):
        # Construim lista de coordonate pentru plotare: Depozit -> Clienti -> Depozit
        xs = [depot.x] + [c.x for c in route] + [depot.x]
        ys = [depot.y] + [c.y for c in route] + [depot.y]

        c = colors[idx % len(colors)]  # Alegem o culoare din lista (ciclic)
        plt.plot(xs, ys, marker='o', color=c, linestyle='-', linewidth=2, label=f'Masina {idx + 1}')

        # Adaugam ID-urile clientilor pe harta pentru claritate
        for client in route:
            plt.annotate(str(client.id), (client.x, client.y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("Optimizarea rutelor finale CVRP")
    plt.legend()
    plt.grid(True)
    plt.show()


# Decodare si evaluare
def decode_and_evaluate(chromosome: List[int], depot: Node, clients: List[Node]):
    """
    Transforma cromozomul (lista simpla de clienti) in rute valide
    si calculeaza costul/fitness-ul.
    Foloseste o metoda Greedy: umple vehiculul curent cat poate, apoi trece la urmatorul.
    """
    routes = []
    current_route = []
    current_load = 0
    client_map = {c.id: c for c in clients}  # Mapare rapida ID -> Obiect Client

    # Decodare: Impartirea secventei in rute pe baza capacitatii
    for client_id in chromosome:
        client = client_map[client_id]
        # Verificam daca clientul incape in camionul curent
        if current_load + client.demand <= VEHICLE_CAPACITY:
            current_route.append(client)
            current_load += client.demand
        else:
            # Daca nu incape, inchidem ruta curenta si incepem una noua
            if current_route:
                routes.append(current_route)
            current_route = [client]
            current_load = client.demand

    # Adaugam ultima ruta ramasa (daca exista)
    if current_route:
        routes.append(current_route)

    # Calcul costuri
    unpenalized_cost = 0.0
    capacity_violation = 0.0

    for route in routes:
        # Distanta Depozit -> Primul client
        dist = euclidean_distance(depot, route[0])
        load = 0

        # Distante intre clienti
        for i in range(len(route) - 1):
            dist += euclidean_distance(route[i], route[i + 1])
            load += route[i].demand

        load += route[-1].demand
        # Distanta Ultimul client -> Depozit
        dist += euclidean_distance(route[-1], depot)

        unpenalized_cost += dist

        # Calculam daca s-a depasit capacitatea (pentru siguranta, desi decodarea previne asta partial)
        if load > VEHICLE_CAPACITY:
            capacity_violation += (load - VEHICLE_CAPACITY)

    # Verificam daca am folosit mai multe vehicule decat avem in flota
    excess_vehicles = max(0, len(routes) - MAX_VEHICLES)

    # Costul penalizat adauga "amenzi" mari pentru incalcarea regulilor
    penalized_cost = unpenalized_cost + (ALPHA * capacity_violation) + (BETA * excess_vehicles)

    # Fitness-ul este inversul costului (cost mic => fitness mare)
    fitness = 1.0 / (1.0 + penalized_cost)

    return fitness, unpenalized_cost, penalized_cost, routes


# Selectie
def selection_roulette(population):
    """
    Selecteaza un parinte folosind metoda Ruletei.
    Indivizii cu fitness mai mare au sanse mai mari sa fie alesi.
    """
    total_fitness = sum(ind.fitness for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += ind.fitness
        if current > pick:
            return ind
    return population[-1]  # Fallback in caz de erori de rotunjire


# Crossover
def crossover_cut_point(parent1, parent2):
    """
    Operator de incrucisare (Order Crossover adaptat).
    Pastreaza o secventa din primul parinte si completeaza cu restul din al doilea,
    pastrand ordinea relativa si evitand duplicatele.
    """
    size = len(parent1.chromosome)
    # Generam 2 puncte de taiere aleatorii pentru a defini un segment
    c1 = random.randint(0, size - 2)
    c2 = random.randint(c1 + 1, size - 1)

    # Segmentul copiat direct de la Parintele 1
    child_p1_segment = parent1.chromosome[c1:c2]

    # Elementele ramase le luam din Parintele 2 (doar cele care nu sunt deja in segment)
    child_rest = [item for item in parent2.chromosome if item not in child_p1_segment]

    # Construim copilul
    child_chromo = []
    rest_idx = 0
    for i in range(size):
        if c1 <= i < c2:
            # In interiorul punctelor de taiere, punem segmentul din P1
            child_chromo.append(child_p1_segment[i - c1])
        else:
            # In afara, completam cu elementele din P2
            child_chromo.append(child_rest[rest_idx])
            rest_idx += 1

    return Individual(chromosome=child_chromo)


# Mutatie
def mutate(individual, current_stagnation):
    """
    Introduce modificari aleatorii in individ pentru a mentine diversitatea.
    Logica este adaptiva: daca algoritmul stagneaza, creste probabilitatea de mutatie.
    """
    base_swap_prob = 0.7  # Probabilitate de baza pentru Swap
    mutation_multiplier = 1.0

    # Daca solutia nu s-a imbunatatit de 20 de generatii, triplam rata de mutatie
    if current_stagnation > 20:
        mutation_multiplier = 3.0

    # Decidem daca facem o mutatie pe acest individ
    if random.random() < (0.2 * mutation_multiplier):
        r = random.random()
        if r < base_swap_prob:
            # Swap Mutation: Schimba locul a doi clienti aleatorii
            idx1, idx2 = random.sample(range(len(individual.chromosome)), 2)
            individual.chromosome[idx1], individual.chromosome[idx2] = individual.chromosome[idx2], \
                individual.chromosome[idx1]
        else:
            # Inversion Mutation: Inverseaza ordinea unei secvente de clienti (ca 2-opt)
            idx1, idx2 = sorted(random.sample(range(len(individual.chromosome)), 2))
            individual.chromosome[idx1:idx2 + 1] = individual.chromosome[idx1:idx2 + 1][::-1]


# Algoritmul principal
def run_genetic_algorithm():
    # 1. Generare problema (depozit si clienti)
    depot, clients = generate_problem(NUM_CLIENTS, 10)

    population = []
    initial_ids = [c.id for c in clients]

    # 2. Initializare populatie (Solutii aleatorii la start)
    for _ in range(POPULATION_SIZE):
        perm = copy.deepcopy(initial_ids)
        random.shuffle(perm)
        ind = Individual(chromosome=perm)
        # Evaluam fiecare individ initial
        f, uc, pc, r = decode_and_evaluate(ind.chromosome, depot, clients)
        ind.fitness = f
        ind.total_cost = uc
        ind.penalized_cost = pc
        ind.routes = r
        population.append(ind)

    # Determinam cea mai buna solutie initiala
    best_global = max(population, key=lambda x: x.fitness)
    stagnation_counter = 0  # Contor pentru generatiile fara imbunatatire
    history_costs = []

    print(f"Cel mai bun cost initial: {best_global.total_cost: .2f}")

    # 3. Bucla evolutiva (Generatii)
    for gen in range(GENERATIONS):
        offspring = []

        # Generam copii pana umplem lista de offspring (lambda)
        while len(offspring) < OFFSPRING_SIZE:
            # Selectie parinti
            p1 = selection_roulette(population)
            p2 = selection_roulette(population)

            # Incrucisare (Crossover)
            child = crossover_cut_point(p1, p2)

            # Mutatie (poate modifica copilul)
            mutate(child, stagnation_counter)

            # Evaluare copil nou creat
            f, uc, pc, r = decode_and_evaluate(child.chromosome, depot, clients)
            child.fitness = f
            child.total_cost = uc
            child.penalized_cost = pc
            child.routes = r
            offspring.append(child)

        # 4. Supravietuire (Strategia mu + lambda)
        # Sortam populatia curenta
        population.sort(key=lambda x: x.fitness, reverse=True)
        elites = population[:ELITISM_COUNT]  # Salvam elita

        # Combinam parintii vechi cu copiii noi
        pool = population + offspring

        # Sortam tot pool-ul si pastram doar cei mai buni 'mu' indivizi
        pool.sort(key=lambda x: x.fitness, reverse=True)
        population = pool[:POPULATION_SIZE]

        # Reintroducem elita daca s-a pierdut (de siguranta)
        if ELITISM_COUNT > 0 and population[0].fitness < elites[0].fitness:
            population[-1] = elites[0]

        # Monitorizare progres
        current_best = population[0]
        history_costs.append(current_best.total_cost)

        if current_best.fitness > best_global.fitness:
            best_global = copy.deepcopy(current_best)
            stagnation_counter = 0  # Resetam contorul daca gasim ceva mai bun
            print(f"Generatia {gen}: Cost mai bun = {best_global.total_cost: .2f} (Masini: {len(best_global.routes)})")
        else:
            stagnation_counter += 1  # Incrementam daca nu gasim nimic mai bun

    # Rezultate finale
    print()
    print(f"REZULTAT FINAL")
    print(f"Cel mai bun cost gasit: {best_global.total_cost: .2f}")
    print(f"Numar vehicule folosite: {len(best_global.routes)}")
    print("Rute detaliate:")
    for idx, r in enumerate(best_global.routes):
        ids = [n.id for n in r]
        load = sum(n.demand for n in r)
        print(f"  Masina {idx + 1}: {ids} (Incarcatura: {load}/{VEHICLE_CAPACITY})")

    # Grafic convergenta
    plt.plot(history_costs)
    plt.xlabel("Generatie")
    plt.ylabel("Cost (Distanta)")
    plt.title("Convergenta Algoritmului Genetic CVRP")
    plt.show()

    # Harta rute
    plot_routes(depot, best_global.routes)


if __name__ == "__main__":
    run_genetic_algorithm()
