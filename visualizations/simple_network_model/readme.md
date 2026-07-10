## A simple network model

Inspired by what Kim sugggested. Throw away all the complications of strengths, trading, etc. We only consider alliances and wars.

Each state can be represented as a node on a graph. Two states that are allies have an edge between them.

The dynamics comprise of two timescales:

1. Making alliances: Two random nodes that aren't linked randomly gain a link.
2. War: Two nodes that are not allies randomly go to war with each other. When at war, we assume the allies (somehow) decide to get involved, but ignore the details. The winner 'conqeurs' the loser, and sets up a puppet government. A random node is chosen as the winner, and the loser has all its alliances removed, and becomes allied with the winner.


### Mean field

If we have a $N$ nodes with $m$ edges, and the connectivity $k$ is $2m/N$. The dynamics are

- Alliance: $m\to m+1$
- War: $m \to m-k + 1$

If the ratio of alliance / war is $r$, we can consider one cycle of $r$ alliances, followed by a war. Then,

$$m_\text{new} = \underbrace{m}_\text{previous edges} + \underbrace{r(1)}_\text{new alliances} +  \underbrace{(-k+1)}_\text{a war}  $$

In the steady state, $m_\text{new} = m = m^\star$:

$$ k^\star = r + 1 $$

The average connectivity is basically set by the ratio. We can calculate $m^\star$ to be

$$ \frac{2m^\star}N = r + 1 $$
$$ m^\star = \frac{(r+1)N}2$$


The percolation threshold (where everyone is connected and the system goes to the densely connected absorbing state) is when the number of nodes $m=N(N-1)/2$. Substituting:

$$ \frac{(r+1)N}{2} = \frac{N(N-1)}2 $$
$$ r+1 = N-1 $$
$$r = N-2$$

### Other models:

1. Preferential attachment: When making an alliance, choose a random node, and have that node prefer to ally with stronger nodes.
2. Targeting the weak: When declaring war, choose a random node, and have them preferentially declare war with weak nodes.