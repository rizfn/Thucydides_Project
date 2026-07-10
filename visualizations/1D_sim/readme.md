### A simple 1D alliance forming and warmongering model.

The algorithm is as follows:

1. Each site on the lattice generates wealth, with some "harvest noise" (rainfall, etc) making some plots more productive than others in the same timestep

2. The new soldiers gained is equal to the amount of wealth at some time 'tau' earlier, i.e, a time delay of the wealth.

3. Every timestep, I choose one random state to do "politics". They look at their (+allies) aggregated strength, and compare that to their neighbours.

    - If the neighbours are weaker than you (lower S) but richer (more W), they'll be stronger in the future, so you go to war (thucydides).

    - If the neighbours are all stronger than you, then you propose an alliance to a random state. That state sees if they need help (if their neighbours are stronger than they are) and if so, accepts. If not, they reject the alliance. This prevents forming one big alliance, as the "winner" refuses to ally with weaklings he doesn't need

    - If the neighbours are weaker (lower S) and less money (lower W), you don't need the alliance, and so you leave it (perhaps unrealistic?).
    
4. Before a war starts, allies are asked sequentially if they want to participate. You participate if it seems likely your side will win. All participants in a war sacrifice some troops for participating, and the winner steals a site from the loser (thus, sacrificing present soldiers "S" for wealth "W", which enables future soldier production)

5. States can never go extinct: if you defeat the last site of a state, they become your "vassal": you destroy their relationship with their existing alliance, and force them to join your alliance.

It freezes, where you have large states and tiny states pushed to size 1. Those function as domain walls, and prevent dynamical behavior. The [2D simulation](../2D_sim/) is more interesting!