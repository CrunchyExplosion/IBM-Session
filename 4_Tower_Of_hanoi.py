def tower_of_hanoi(n, source, auxillary, destination):
    if n == 1:
      print(f"Move disk 1 from {source} to {destination}")
      return

    tower_of_hanoi(n - 1, source, destination, auxillary)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n - 1, auxillary, source, destination)
n = 4
tower_of_hanoi(n,'A','B','C')
