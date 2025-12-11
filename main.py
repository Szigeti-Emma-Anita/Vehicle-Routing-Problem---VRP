import math
import random
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# CONFIGURARE
NUM_CLIENTS = 40       # Numar clienti (fara depot; nodul 0 este depozitul)
VEHICLE_CAPACITY = 20  # Q (Capacitatea maxima de incarcare a fiecarui vehicul)
MAX_VEHICLES = 20      # K (Numarul maxim de vehicule disponibile in flota)
GENERATIONS = 200      # Numar generatii (cat timp ruleaza algoritmul)
POPULATION_SIZE = 50   # mu (Marimea populatiei de solutii parinti)
OFFSPRING_SIZE = 40    # lambda (Numarul de solutii copii generate per generatie)
ELITISM_COUNT = 1      # E (Numarul celor mai buni indivizi care trec automat in generatia urmatoare)

# Coeficienti penalizare
ALPHA = 1000.0  # Penalizare depasire capacitate (valoare mare pentru a forta respectarea limitei Q)
BETA = 1000.0   # Penalizare vehicule in exces (valoare mare pentru a nu folosi mai mult de K vehicule)


# STRUCTURI DE DATE
@dataclass
class Node:
    """
    Reprezinta un punct pe harta.
    Poate fi Depozitul (id=0) sau un Client (id>0).
    """
    id: int
    x: float
    y: float
    demand: int  # Cererea clientului (cat trebuie livrat)


@dataclass
class Individual:
    """
    Reprezinta o solutie posibila in Algoritmul Genetic.
    
    Atribute:
        chromosome: Lista de ID-uri clienti (permutare unica). Aici ne asiguram ca nu exista duplicate.
        fitness: Scorul solutiei (mai mare = mai bun).
        total_cost: Distanta totala a rutelor.
        penalized_cost: Costul care include penalizarile (folosit pentru selectie).
        routes: Listele efective de noduri, grupate pe vehicule.
    """
    chromosome: List[int]
    fitness: float = 0.0
    total_cost: float = 0.0
    penalized_cost: float = 0.0
    routes: List[List[Node]] = None


# GENERARE DATE
def generate_problem(num_clients: int, max_demand: int) -> Tuple[Node, List[Node]]:
    """
    Genereaza o instanta aleatorie a problemei CVRP.
    
    Returneaza:
        depot: Nodul depozit (fixat in centru).
        clients: Lista de obiecte Node reprezentand clientii.
    """
    depot = Node(0, 50, 50, 0)  # Depozitul este plasat in centrul hartii
    clients = []
    for i in range(1, num_clients + 1):
        clients.append(Node(
            i,
            random.uniform(0, 100),       # Coordonata X aleatorie
            random.uniform(0, 100),       # Coordonata Y aleatorie
            random.randint(1, max_demand) # Cerere aleatorie
        ))
    return depot, clients


def euclidean_distance(n1: Node, n2: Node) -> float:
    """ Calculeaza distanta in linie dreapta intre doua noduri. """
    return math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)


# VIZUALIZARE
def plot_routes(depot: Node, routes: List[List[Node]]):
    """
    Deseneaza harta finala folosind matplotlib.
    Fiecare vehicul va avea o culoare diferita.
    """
    colors = ['blue', 'green', 'red', 'cyan', 'pink', 'yellow', 'black', 'orange', 'purple', 'brown']
    plt.figure(figsize=(10, 8))

    # 1. Desenam Depozitul
    plt.plot(depot.x, depot.y, 'rs', markersize=15, label='Depozit')

    # 2. Desenam fiecare ruta
    for idx, route in enumerate(routes):
        # Construim coordonatele: Depot -> Client 1 -> ... -> Client N -> Depot
        xs = [depot.x] + [c.x for c in route] + [depot.x]
        ys = [depot.y] + [c.y for c in route] + [depot.y]

        c = colors[idx % len(colors)]
        plt.plot(xs, ys, marker='o', color=c, linestyle='-', linewidth=2, label=f'Vehicul {idx + 1}')

        # Adaugam ID-urile pe harta
        for client in route:
            plt.annotate(str(client.id), (client.x, client.y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("Optimizarea rutelor finale CVRP")
    plt.legend()
    plt.grid(True)
    plt.show()


# DECODARE SI EVALUARE
def decode_and_evaluate(chromosome: List[int], depot: Node, clients: List[Node]):
    """
    Transforma cromozomul (lista simpla de clienti) in rute valide.
    
    Mecanism:
        Parcurge lista de clienti unul cate unul.
        Adauga clientul in vehiculul curent.
        Daca se depaseste capacitatea (VEHICLE_CAPACITY), se inchide ruta curenta
        si se deschide una noua pentru vehiculul urmator.
    
    Deoarece 'chromosome' este o permutare a tuturor ID-urilor,
    aceasta functie garanteaza ca fiecare client este vizitat exact o data.
    """
    routes = []
    current_route = []
    current_load = 0
    client_map = {c.id: c for c in clients}  # Mapare rapida ID -> Obiect

    for client_id in chromosome:
        client = client_map[client_id]
        if current_load + client.demand <= VEHICLE_CAPACITY:
            current_route.append(client)
            current_load += client.demand
        else:
            if current_route:
                routes.append(current_route)
            current_route = [client]
            current_load = client.demand

    if current_route:
        routes.append(current_route)

    # Calcul costuri
    unpenalized_cost = 0.0
    capacity_violation = 0.0

    for route in routes:
        dist = euclidean_distance(depot, route[0])
        load = 0
        for i in range(len(route) - 1):
            dist += euclidean_distance(route[i], route[i + 1])
            load += route[i].demand

        load += route[-1].demand
        dist += euclidean_distance(route[-1], depot)
        unpenalized_cost += dist
        
        # Verificare suplimentara (decodarea de mai sus previne depasirea, dar o pastram pt robustete)
        if load > VEHICLE_CAPACITY:
            capacity_violation += (load - VEHICLE_CAPACITY)

    excess_vehicles = max(0, len(routes) - MAX_VEHICLES)
    
    # Formula cost penalizat
    penalized_cost = unpenalized_cost + (ALPHA * capacity_violation) + (BETA * excess_vehicles)
    
    # Fitness invers proportional cu costul
    fitness = 1.0 / (1.0 + penalized_cost)

    return fitness, unpenalized_cost, penalized_cost, routes


# SELECTIE
def selection_roulette(population: List[Individual]) -> Individual:
    """
    Roulette Wheel Selection
    Selecteaza un parinte proportional cu fitness-ul sau.
    """
    total_fitness = sum(ind.fitness for ind in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += ind.fitness
        if current > pick:
            return ind
    return population[-1]


# CROSSOVER (Esential pentru prevenirea duplicatelor)
def crossover_cut_point(parent1: Individual, parent2: Individual) -> Individual:
    """
    Realizeaza incrucisarea intre doi parinti pentru a crea un copil.
    
    Mecanism ANTI-DUPLICATE:
    1. Se taie un segment aleatoriu din Parintele 1.
    2. Acest segment este copiat direct in Copil, la aceeasi pozitie.
    3. Pozitiile ramase goale in copil sunt completate cu genele din Parintele 2.
    4. FOARTE IMPORTANT: Se copiaza din Parintele 2 doar genele care NU exista deja in segment.
    
    Acest lucru asigura ca fiecare ID de client apare o singura data in copil.
    """
    size = len(parent1.chromosome)
    c1 = random.randint(0, size - 2)
    c2 = random.randint(c1 + 1, size - 1)

    # Segmentul fixat din primul parinte
    child_p1_segment = parent1.chromosome[c1:c2]

    # Pregatim restul genelor din parintele 2
    # Filtrul "if item not in child_p1_segment" previne duplicatele
    child_rest = [item for item in parent2.chromosome if item not in child_p1_segment]

    child_chromo = []
    rest_idx = 0
    
    # Reconstruim cromozomul complet
    for i in range(size):
        if c1 <= i < c2:
            # In zona taiata, punem genele din P1
            child_chromo.append(child_p1_segment[i - c1])
        else:
            # In afara, completam cu genele din P2
            child_chromo.append(child_rest[rest_idx])
            rest_idx += 1

    return Individual(chromosome=child_chromo)


# MUTATIE
def mutate(individual: Individual, current_stagnation: int):
    """
    Aplica modificari aleatorii asupra unui individ.
    
    Atat Swap cat si Inversion doar REARANJEAZA clientii existenti.
    Ele nu pot crea duplicate si nu pot sterge clienti.
    """
    base_swap_prob = 0.7
    mutation_multiplier = 1.0

    # Adaptivitate: daca algoritmul s-a blocat (stagneaza), creste sansa de mutatie
    if current_stagnation > 20:
        mutation_multiplier = 3.0

    if random.random() < (0.2 * mutation_multiplier):
        r = random.random()
        if r < base_swap_prob:
            # Swap: Interschimba doi clienti
            idx1, idx2 = random.sample(range(len(individual.chromosome)), 2)
            individual.chromosome[idx1], individual.chromosome[idx2] = individual.chromosome[idx2], individual.chromosome[idx1]
        else:
            # Inversion: Inverseaza o secventa
            idx1, idx2 = sorted(random.sample(range(len(individual.chromosome)), 2))
            individual.chromosome[idx1:idx2 + 1] = individual.chromosome[idx1:idx2 + 1][::-1]


# ALGORITMUL PRINCIPAL
def run_genetic_algorithm():
    # 1. Initializare date problema
    depot, clients = generate_problem(NUM_CLIENTS, 10)

    # VERIFICARE CONSTRANGERE FLOTA
    total_demand = sum(c.demand for c in clients)
    max_fleet_capacity = MAX_VEHICLES * VEHICLE_CAPACITY
    min_vehicles_needed = math.ceil(total_demand / VEHICLE_CAPACITY)

    print(f"Cerere totala clienti: {total_demand}")
    print(f"Capacitate flota: {max_fleet_capacity}")
    print(f"Numar minim vehicule necesare: {min_vehicles_needed}")

    if total_demand > max_fleet_capacity:
        print(f"\nEROARE: Cererea totala ({total_demand}) depaseste capacitatea flotei ({max_fleet_capacity}).")
        print(f"Trebuie sa maresti numarul de vehicule (acum {MAX_VEHICLES}) sau capacitatea.")
        return # Opreste executia pentru a nu da rezultate eronate
    
    if min_vehicles_needed > MAX_VEHICLES:
         print(f"\nATENTIE: Este matematic imposibil sa servesti toti clientii cu {MAX_VEHICLES} vehicule.")
         return

    # 2. Creare populatie initiala
    population = []
    initial_ids = [c.id for c in clients] # Lista [1, 2, ..., N]

    for _ in range(POPULATION_SIZE):
        perm = copy.deepcopy(initial_ids)
        random.shuffle(perm) # Amestecare aleatorie -> Permutare unica -> Fara duplicate
        ind = Individual(chromosome=perm)
        
        f, uc, pc, r = decode_and_evaluate(ind.chromosome, depot, clients)
        ind.fitness = f
        ind.total_cost = uc
        ind.penalized_cost = pc
        ind.routes = r
        population.append(ind)

    best_global = max(population, key=lambda x: x.fitness)
    stagnation_counter = 0
    history_costs = []

    print(f"Cel mai bun cost initial: {best_global.total_cost:.2f}")

    # 3. Bucla Generationala
    for gen in range(GENERATIONS):
        offspring = []
        
        # Generare copii noi
        while len(offspring) < OFFSPRING_SIZE:
            p1 = selection_roulette(population)
            p2 = selection_roulette(population)

            # Aici se intampla magia anti-duplicate
            child = crossover_cut_point(p1, p2)
            mutate(child, stagnation_counter)

            f, uc, pc, r = decode_and_evaluate(child.chromosome, depot, clients)
            child.fitness = f
            child.total_cost = uc
            child.penalized_cost = pc
            child.routes = r
            offspring.append(child)

        # 4. Supravietuire (mu + lambda)
        # Sortam populatia curenta descrescator dupa fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Salvam elita (cel mai bun individ)
        elites = population[:ELITISM_COUNT]

        # Punem la comun parintii si copiii
        pool = population + offspring
        pool.sort(key=lambda x: x.fitness, reverse=True)
        
        # Selectam cei mai buni indivizi pentru urmatoarea generatie
        population = pool[:POPULATION_SIZE]

        # Reintroducem elita daca cumva a fost pierduta (de siguranta)
        if ELITISM_COUNT > 0 and population[0].fitness < elites[0].fitness:
            population[-1] = elites[0]

        # Monitorizare
        current_best = population[0]
        history_costs.append(current_best.total_cost)

        if current_best.fitness > best_global.fitness:
            best_global = copy.deepcopy(current_best)
            stagnation_counter = 0
            print(f"Generatia {gen}: Cost mai bun = {best_global.total_cost:.2f} (Vehicule: {len(best_global.routes)})")
        else:
            stagnation_counter += 1

    # REZULTATE FINALE
    print(f"\nREZULTAT FINAL")
    print(f"Cel mai bun cost gasit: {best_global.total_cost:.2f}")
    print(f"Numar vehicule folosite: {len(best_global.routes)}")
    print("Rute detaliate:")
    for idx, r in enumerate(best_global.routes):
        ids = [n.id for n in r]
        load = sum(n.demand for n in r)
        print(f"  Vehicul {idx + 1}: {ids} (Incarcatura: {load}/{VEHICLE_CAPACITY})")

    # Plotare grafic cost
    plt.plot(history_costs)
    plt.xlabel("Generatie")
    plt.ylabel("Cost (Distanta)")
    plt.title("Convergenta Algoritmului Genetic pentru CVRP")
    plt.show()

    # Plotare harta finala
    plot_routes(depot, best_global.routes)


if __name__ == "__main__":
    run_genetic_algorithm()