\documentclass[journal]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\newtheorem{remark}{Remark}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{multirow}
\usepackage{appendix}
\usepackage{verbatim}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}


\graphicspath{ {./figures/} }

\begin{document}

\title{Stress-Sharing: A Bio-Inspired Approach to Decentralized Fault Repair in Modular Spacecraft}

\author{Sidhdharth D.~Sikka, Yue~Shen, and Shaoshuai~Mou%
\thanks{Sidhdharth D. Sikka is with Manifold Research Group and the School of Aeronautics and Astronautics, Purdue University, West Lafayette, IN 47907 USA (e-mail: sikkas@purdue.edu).}%
\thanks{Yue Shen is with Manifold Research Group (e-mail: yzs.shen@gmail.com).}%
\thanks{Shaoshuai Mou is with the School of Aeronautics and Astronautics, Purdue University, West Lafayette, IN 47907 USA (e-mail: mous@purdue.edu).}%
}

\markboth{IEEE Robotics and Automation Letters,~Vol.~X, No.~X, Month~2026}%
{Sikka \MakeLowercase{\textit{et al.}}: Damage-Responsive Reconfiguration}


\maketitle

\begin{abstract}
Structural damage in modular spacecraft can disrupt mechanical and communication connectivity, reducing system capability. Existing approaches rely on redundancy or preplanned reconfiguration and do not enable autonomous repair under local information and physical constraints. We model the spacecraft as a graph of interchangeable modules and formulate repair as restoring active connectivity while preserving the original structure. We introduce a fully decentralized two-phase reconfiguration strategy inspired by biological stress sharing. During coagulation, local failure signals propagate through the structure and non-critical modules move toward damaged regions; during reformation, modules recover the prior shape using local neighbor memory. Monte Carlo simulations across randomized damage scenarios show reliable connectivity restoration with high structural similarity to the original configuration, without global planning or centralized control. These results demonstrate that distributed bio-inspired policies can enable resilient long duration modular spacecraft.
\end{abstract}

\begin{IEEEkeywords}
Modular robotics, self-reconfiguration, fault tolerance, bio-inspired robotics, morphogenesis, stress-sharing, resilient control, multi-agent systems
\end{IEEEkeywords}


\section{Introduction}

Space systems operate in environments where a single failure can compromise an entire mission, and repair or replacement is typically impossible. Current mitigation strategies rely on redundancy and over-engineering, increasing mass and cost while remaining vulnerable to unforeseen damage or cascading failures. Modular robotic spacecraft offer an alternative: systems capable of physically reorganizing themselves to maintain functionality after damage.

\begin{figure}[h]
\includegraphics[scale=0.21]{reconfiguration.png}
\caption{Homeostatic reconfiguration in a modular robotic system.
After a localized failure event, neighboring modules reposition to
restore connectivity in the underlying interaction graph. The process
models a distributed response to electrical or communication loss
without centralized coordination.}
\label{fig:reconfig}
\end{figure}

Recent work has advanced both hardware and algorithms for modular reconfiguration. ElectroVoxel demonstrated autonomous pivot-based reconfiguration in microgravity~\cite{shay2022electrovoxel}. Universal reconfiguration using a constant number of helper modules was shown in~\cite{akitaya2019}. Three-dimensional lattice systems such as M-Blocks enable flywheel-driven pivots~\cite{romanishin2015}, and decentralized assembly strategies have been explored for modular satellites~\cite{li2021}. These approaches primarily target global shape transformation or universality, typically assuming continuous connectivity and sufficient planning time.

Under structural damage, these assumptions fail: connectivity may be lost, information propagation restricted, and global reconfiguration computationally or physically infeasible. Instead, we focus on \emph{homeostatic reconfiguration}, where modules execute rapid local actions to restore operational capability, as illustrated in Fig.~\ref{fig:reconfig}.

In this work we propose a decentralized damage-responsive reconfiguration algorithm for modular space robots. Repair is formulated as restoring connectivity while preserving structure in a lattice-constrained system. Directional stress-sharing signals guide connectivity-safe pivot motions, followed by a reformation phase that recovers prior module positions. The method operates under local sensing and no centralized planning.

\subsection{Related Work}
Connectivity maintenance has been studied extensively in wireless networks. RECRA~\cite{imran2012} restores connectivity by relocating non-critical nodes using 1-hop information, while DCRMF~\cite{zhang2018} extends this approach to multiple failures using distributed state machines and 2-hop neighborhood data. However, these methods address communication topology only and do not consider geometric constraints, fixed unit-distance connections, or structural load transfer required in modular robotic systems.

Formation control literature emphasizes redundancy and robustness~\cite{dinitz2011}, typically switching to backup configurations or removing failed agents rather than reorganizing structure. Biological inspiration has motivated regenerative modular robotics, including robotic stem cells~\cite{rubenstein2009} and hormone-based repair~\cite{yang2017self}. These works demonstrate adaptive behavior but do not address restoring load-bearing connectivity under strict physical constraints.

To our knowledge, no prior framework combines autonomous damage detection, connectivity evaluation, and physically feasible reconfiguration using only local sensing and computation in space-relevant conditions.

The remainder of this paper is organized as follows: Section II establishes the graph-theoretic preliminaries and physical constraints of the modular system; Section III formulates the homeostatic reconfiguration problem; Section IV details the proposed distributed stress-sharing and two-phase repair algorithms; Section V presents Monte Carlo simulation results and performance analysis; and Section VI provides concluding remarks and directions for future research.

\section{Preliminaries}

We model a modular spacecraft as a lattice-connected assembly of interchangeable modules, where we assume only the relative position of the modules is relevant and not their relative attitudes. At time $t$, the system is represented as a directed graph
\[
\mathcal{G}_t = (V, E_t, g_t),
\]
where $V$ is the set of modules (vertices), 
$E_t \subseteq V \times V$ is the set of directed edges representing physical connections between modules, 
and
\[
g_t : E_t \to \mathrm{SE}(3)
\]
assigns to each directed edge $(u,v)$ the relative rigid-body transformation from module $u$ to module $v$. Self-loops are excluded from the network topology, as they lack physical realization within a modular spacecraft architecture.

In this work, we restrict edge transformations to pure translations, so that for each $(u,v) \in E_t$,
\[
g_t(u,v) = (I, \boldsymbol{\rho}_{uv}), \qquad \boldsymbol{\rho}_{uv} \in \mathbb{R}^3,
\]
with inverse
\[
g_t(v,u) = g_t(u,v)^{-1} = (I, -\boldsymbol{\rho}_{uv}).
\]

Relative poses are composed along paths in the graph using the group operation in $\mathrm{SE}(3)$. 
Because all edge transformations are pure translations, composition along any path reduces to vector addition of the translation components. 
In particular, for a path $u \rightarrow v \rightarrow w$,
\[
g_t(u,w) = g_t(u,v) \circ g_t(v,w)
       = (I, \boldsymbol{\rho}_{uv} + \boldsymbol{\rho}_{vw}).
\]

Each relative translation $\boldsymbol{\rho}_{uv}$ is constrained to lie along the unit lattice directions
\[
\boldsymbol{\rho}_{uv}\in\{\pm \hat{\boldsymbol{x}},\, \pm \hat{\boldsymbol{y}},\, \pm \hat{\boldsymbol{z}}\},
\qquad \|\boldsymbol{\rho}_{uv}\| = 1.
\]

\subsection{Active Subgraph and Damage}

A subset $\bar{V}_t \subseteq V$ denotes the set of \emph{active} modules at time $t$.
The corresponding active edge set is $\bar{E}_t \subseteq \bar{V}_t \times \bar{V}_t$, and the active subgraph is defined as
\[
\bar{\mathcal{G}}_t = (\bar{V}_t,\, \bar{E}_t,\, \bar{g}_t),
\]
where $\bar{g}_t$ denotes the restriction of the edge transformation map to the active edges. Only edges in $\bar{E}_t$ participate in reconfiguration and control.

A damage event at time $t$ induces a new active set
\[
\bar{V}_{t+1} = \bar{V}_t \setminus F_t,
\]
where $F_t \subseteq \bar{V}_t$ is the set of damaged or failed modules at that time.
All physical connections remain present in the underlying graph $E_t$, but edges incident to inactive vertices are excluded from $\bar{E}_{t+1}$.
As a result, the active subgraph $\bar{\mathcal{G}}_{t+1}$ may become disconnected, necessitating reconfiguration to restore structural or functional connectivity. The neighbor set for a vertex $u\in V$ is defined as $N_t(u) := \{\, w \in V \mid (u,w) \in E_t \,\}$, and the \textit{active} neighbor set for a vertex $u\in V$ is defined as $\bar{N}_t(u) := \{\, w \in \bar{V}_\tau \mid (u,w) \in \bar{E}_\tau \,\}$.

\subsection{Detachability and Attachability}
\label{sec:detachattach}

A vertex $u \in \bar{V}_t$ may participate in reconfiguration subject to the
following rules.

\paragraph{Detachability}
A vertex $u \in \bar{V}_t$ may detach from a neighboring vertex
$v \in V$ by removing both directed edges $(u,v)$ and $(v,u)$ from $E_t$,
resulting in
\[
E_{t+1}
\;=\;
E_t \setminus \{(u,v),(v,u)\},
\]
if and only if
\[
u \in \bar{V}_t
\qquad \text{and} \qquad
|\bar{N}_{t+1}(u)| \ge 1.
\]
That is, the vertex initiating detachment must remain connected to at
least one other active neighbor after the operation. The vertex $v$ may
be either active or inactive.

\paragraph{Attachability}
A vertex $u \in \bar{V}_t$ may attach to a vertex $v \in V$ by adding
directed edges $(u,v)$ and $(v,u)$ to the graph,
\[
E_{t+1}
\;=\;
E_t \cup \{(u,v),(v,u)\},
\]
if and only if the following conditions hold:
\begin{enumerate}
  \item $u,v \in \bar{V}_t$,
  \item the relative displacement
  \[
  \boldsymbol{\rho}_{uv}
  \;=\;
  p_t(v) - p_t(u)
  \;\in\;
  \{\pm \hat{\boldsymbol{x}},\, \pm \hat{\boldsymbol{y}},\, \pm \hat{\boldsymbol{z}}\},
  \]
  and
  \item the direction $\boldsymbol{\rho}_{uv}$ is not already occupied by an existing edge incident to $v$, i.e.,
  \[
  g_{t+1}(u,v) \neq g_t(w,v)
  \qquad
  \forall w \in N_t(v).
  \]
\end{enumerate}
When the attachment occurs, the edge transformations are defined as
\[
g_{t+1}(u,v) = (I, \boldsymbol{\rho}_{uv}),
\qquad
g_{t+1}(v,u) = (I, -\boldsymbol{\rho}_{uv}).
\]

\subsection{Shape Similarity}

For an active vertex $u \in \bar{V}_t$, define the set of distances from
$u$ to all other active vertices as
\[
P_t(u)
\;:=\;
\big\{\, \| \boldsymbol{\rho}_{uv} \|_2
\;\big|\;
v \in \bar{V}_t,\; v \neq u \,\big\},
\]
where $\boldsymbol{\rho}_{uv} = p_t(v) - p_t(u)$ denotes the relative
lattice displacement between vertices $u$ and $v$.

Aggregating these sets over all active vertices yields
\[
P_t
\;:=\;
\bigcup_{u \in \bar{V}_t} P_t(u),
\]
which is the set of all pairwise inter-module distances in the active
subgraph.

For two shapes $P, Q$, their difference can be measured as:

$$
\mathrm{diff}(P,Q) = \frac{|P|-|P\cap{Q}|}{|P|}
$$

This asymmetric measure penalizes the loss of pre-damage structural
features and is anchored to the original shape $P$.

\section{Problem Formulation}
\label{sec:problem}

Following a damage event at time $t$, the active subgraph
$\bar{\mathcal{G}}_{t+1}$ may be disconnected.
We seek a sequence of admissible reconfiguration operations that restores
active connectivity while minimizing deviation from the pre-damage
shape.

Let $P_t$ denote the pre-damage shape and $P_{t_{\mathrm{f}}}$ the terminal shape at
time $t_{\mathrm{f}} \ge t+1$, expressed in the same lattice frame.
The reconfiguration problem is formulated as
\[
\min_{\{\bar{\mathcal{G}}_\tau\}_{t+1}^{t_{\mathrm{f}}}}
\quad
\mathrm{diff}\!\left(P_t,\, P_{t_{\mathrm{f}}}\right)
\]
subject to
\begin{align*}
& E_{\tau+1} = E_\tau \setminus \{(u,v),(v,u)\}
\;\Rightarrow\;
\Big(
u \in \bar{V}_\tau
\;\wedge\;
|\bar{N}_{\tau+1}(u)| \ge 1
\Big),
\\[0.75ex]
& E_{\tau+1} = E_\tau \cup \{(u,v),(v,u)\}
\;\Rightarrow\;
\begin{cases}
u, v \in \bar{V}_\tau,\\
g_{\tau+1}(u,v) = (I,\boldsymbol{\rho}_{uv}),\\
\boldsymbol{\rho}_{uv} \in
\{\pm\hat{\boldsymbol{x}},\pm\hat{\boldsymbol{y}},\pm\hat{\boldsymbol{z}}\},\\
g_{\tau+1}(u,v) \neq g_\tau(w,v)
\\ \qquad \qquad \forall w \in N_\tau(v),
\end{cases}
\\[0.5ex]
& \bar{\mathcal{G}}_{t_{\mathrm{f}}} \ \text{is connected.}
\end{align*}

%\begin{remark}[One-Step Connectivity Recovery]
%label{rem:one_step_recovery}
%If the active subgraph $\bar{\mathcal{G}}_{t+1}$ is disconnected and there
%exists a valid single-step transition in $T$ that introduces an active
%edge between two previously distinct connected components of
%$\bar{\mathcal{G}}_{t+1}$, then the resulting active subgraph is connected.
%This observation motivates the use of direct cross-component attachment
%operations for rapid connectivity recovery.
%\end{remark}

\section{Proposed Approach: Distributed Stress-Sharing Repair}
\label{sec:approach}

\subsection{Homeostasis in Nature}
Biological systems provide concrete mechanisms for adaptation to damage. Wound healing in tissue, depicted in Figure \ref{fig:woundrepair}, shows how cells migrate toward an injury site, guided by chemical gradients and mechanical cues, and coordinate through signaling pathways to close gaps. Neural plasticity demonstrates that networks can reorganize to bypass damaged regions and preserve function, as new synaptic connections form and strengthen with use.

\begin{figure}[h]
\includegraphics[scale=0.105]{cellular_migration.png}
\caption{Example of cellular migration during wound healing showing initial tissue damage (left), directional cell migration toward the wound site (middle), and successful tissue repair (right). Tissue repair occurs without centralized oversight, through local mechanisms, such as electrical or chemical gradients and signal propagation.}
\label{fig:woundrepair}
\end{figure}

Structural adaptation presents further examples. Bones remodel through a cycle of resorption and deposition, driven by osteoclasts and osteoblasts responding to mechanical stress. Microfractures are repaired and regions under higher load become reinforced. Plant vascular systems also rely on adaptive rerouting. When vessels in the xylem or phloem are blocked, pressure changes and the growth of new conduits redirect fluid transport, ensuring continuity even when individual channels fail.

Recent work has also emphasized the role of stress sharing in collective systems. Shreesha and Levin~\cite{shreesha2024} argue that when cells exchange information about internal stress states, they more efficiently coordinate morphogenetic processes. Stress sharing operates as a lightweight communication channel: units signal their level of strain, allowing collectives to reorganize adaptively without centralized control.

\subsection{Failure Signals and Directional Propagation}

We propose an asynchronous distributed policy inspired by stress sharing: failure information diffuses through the active subgraph, biasing non-critical modules to migrate toward the fault while preserving connectivity. This coagulation process is detailed in Algorithm \ref{alg:coagulation}. The full proposed approach is summarized in Figure \ref{fig:repair_process}.

\begin{figure*}[!t]
  \centering
  \includegraphics[width=0.80\linewidth]{combined_scaled.png}
  \caption{Two-phase distributed repair process. Left: Coagulation phase restores connectivity after damage. Upon detecting or receiving a distress token, a module propagates a directional estimate of the fault location and selects a candidate movement aligned with that direction. Motion is executed only if a bounded-hop criticality test certifies local mobility and detachment preserves at least one active neighbor. Right: Restructuring performs partial shape recovery using stored pre-damage neighbor sets $N_0(u)$. Modules generate and propagate rendezvous tokens, select a reunion target $(v^\star, \boldsymbol{\zeta}^\star) \in \mathcal{R}_t(u)$, and execute a reformation move under the same mobility and connectivity constraints. Both loops operate asynchronously and repeat until connectivity and adjacency relationships are restored.}
  \label{fig:repair_process}
\end{figure*}


Each ego agent $u$ maintains a
set of \emph{distress tokens}
\[
\mathcal{M}_t(u) \subseteq \bar{V}_t \times \mathbb{R}^3,
\]
where a token $(f,\boldsymbol{\xi})$ indicates a suspected failed vertex
$f \in F_t$ and a direction vector $\boldsymbol{\xi}$ expressed in the frame of $u$.

\paragraph{Local detection}
If $u$ detects that a former neighbor $f$ is inactive (e.g., heartbeat
timeout), then $u$ generates the token
\[
(f,\boldsymbol{\xi}) \qquad
\boldsymbol{\xi} := \boldsymbol{\rho}_{uf},
\]
where the right-hand side is interpreted as the last-known displacement prior to deactivation.

\paragraph{Propagation}
When $u$ receives or creates a token $(f,\boldsymbol{\xi})$ from a neighbor, it
updates and forwards the token by composing the edge translation:
\[
\boldsymbol{\xi} \leftarrow \boldsymbol{\rho}_{wu} + \boldsymbol{\xi},
\]
and broadcasts $(f,\boldsymbol{\xi})$ to neighbors  $w\in\bar{N}_t(u)$.
Because path composition is additive under pure translations,
$\boldsymbol{\xi}$ approximates the displacement from $u$ toward the fault location.

\subsection{Local Criticality Test}
\label{sec:crittest}

Upon receiving (or maintaining) any nonempty $\mathcal{M}_t(u)$, the ego
agent $u$ decides whether it is safe to move without inducing a new
disconnect.

We use a conservative bounded-hop test. Define the two-hop active neighborhood
\[
\bar{N}_t^{(2)}(u) := \bigcup_{v\in \bar{N}_t(u)} \bar{N}_t(v).
\]
The ego agent $u$ is declared \emph{movable} if either
\[
|\bar{N}_t(u)| = 1
\]
(leaf condition), or there exists a neighbor $v \in \bar{N}_t(u)$ such
that
\[
\bar{N}_t^{(2)}(u)\setminus\{v\} \;\cap\; \bar{N}_t(v) \neq \emptyset.
\]
Equivalently, $u$ has at least one neighbor that shares a neighbor with
$u$, providing a local alternative path. This is a sufficient condition
for $u$ to detach from one incident edge without isolating itself under
our constraints. This test is local and \textit{conservative}, and has been used
in prior work on network fault repair
\cite{imran2012, zhang2018}. Even when the test fails, a global
alternative path may still exist.

\subsection{Distress-Directed Pivot Selection}
\label{sec:pivotselection}
Once an agent $u$ is certified movable, it selects a target distress token $(f^\star, \boldsymbol{\xi}^\star) \in \mathcal{M}_t(u)$ that is closest i.e. $$\boldsymbol{\xi}^\star = \argmin_{(f,\boldsymbol{\xi})\in\mathcal{M}_t(u)} \|\boldsymbol{\xi}\|$$ The agent evaluates all physically admissible pivot motions, meaning satisfying the detachability and attachability constraints in \ref{sec:detachattach}, where each candidate motion induces a local lattice displacement $\Delta \boldsymbol{p} \in \mathbb{R}^3$. Let $\langle \cdot\, ,\cdot\rangle$ denote the inner product of two vectors in $\mathbb{R}^3$. To prioritize movement toward the fault, the agent selects the pivot that maximizes alignment with the target direction:
\[
\Delta \boldsymbol{p}^\star = \arg\max_{\Delta \boldsymbol{p}} \langle \Delta \boldsymbol{p}, \boldsymbol{\xi}^\star \rangle,
\]
subject to the \emph{alignment constraint} $\langle \Delta \boldsymbol{p}^\star, \boldsymbol{\xi}^\star \rangle > 0$. If no admissible pivot yields positive alignment, $u$ remains stationary to avoid drifting away from the disconnected component. Otherwise, $u$ executes the optimal pivot and updates its local neighborhood state.


\begin{algorithm}[t]
\caption{Coagulation Policy}
\label{alg:coagulation}
\begin{algorithmic}[1]
\STATE \textbf{def} coagulation$(u, \bar{N}_t(u),N_t(u), \mathcal{M}_t(u))$:
\STATE \hspace{1em} \textbf{while True}$:$
\STATE \hspace{2em} \textbf{if} $f \notin \bar{V}_t$ \textbf{and} $f \in N_t(u)$:
\STATE \hspace{3em} $\boldsymbol{\xi} \gets \boldsymbol{\rho}_{uf}$
\STATE \hspace{3em} $\mathcal{M}_{t+1}(u) \gets \mathcal{M}_t(u) \cup (f,\boldsymbol{\xi})$
\STATE \hspace{2em} \textbf{for each} $(f,\boldsymbol{\xi}) \in \mathcal{M}_t(u)$:
\STATE \hspace{3em} \textbf{for each} $w \in \bar{N}_t(u)$:
\STATE \hspace{4em} $\boldsymbol{\xi} \gets \boldsymbol{\rho}_{wu}+\boldsymbol{\xi}$
\STATE \hspace{4em} $\mathcal{M}_{t+1}(w) \gets \mathcal{M}_t(w) \cup (f,\boldsymbol{\xi})$
\STATE \hspace{2em} \textbf{if} len$(\mathcal{M})=0$:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} movable $\gets$ Criticality Test (Section \ref{sec:crittest})
\STATE \hspace{2em} \textbf{if} \textbf{not} movable:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} $(f^\star,\boldsymbol{\xi}^\star)\gets$ Select Token (Section \ref{sec:pivotselection})
\STATE \hspace{2em} $\Delta \boldsymbol{p}^\star \gets \arg\max_{\Delta \boldsymbol{p}} \langle \Delta \boldsymbol{p}, \boldsymbol{\xi}^\star \rangle$
\STATE \hspace{2em} \textbf{if} $\langle \Delta \boldsymbol{p}^\star, \boldsymbol{\xi}^\star \rangle \le 0$:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} Execute $\Delta \boldsymbol{p}^\star$
\end{algorithmic}
\end{algorithm}

\begin{comment}

\subsection{Restructuring After Coagulation}
\label{sec:reformation}

After the coagulation phase, the network connectivity may be restored but
the original shape is lost. We therefore consider a two-phase strategy:
(i) \emph{coagulation}, in which agents migrate toward faults to restore
connectivity, and (ii) \emph{reformation}, in which agents partially
recover the pre-damage shape without re-breaking newly restored
connectivity. The restructuring approach is summarized in Algorithm \ref{alg:restructuring}.

Let $t_c$ denote the time at which coagulation terminates and reformation
begins. We assume each surviving active vertex
$u \in \bar{V}_{t_c}$ stores a \emph{movement history} accumulated
during the coagulation phase beginning at reference time $t_0 \le t$
prior to fragmentation:
\[
\boldsymbol{p}_0(u) \in \mathbb{R}^3, \qquad
\boldsymbol{d}(u) := \sum_{i=1}^{k_u} \delta_i(u),
\]
where $\boldsymbol{p}_0(u)$ is the pre-damage position, $\delta_i(u)$ is
the translation vector of the $i$-th coagulation move, and
$\boldsymbol{d}(u)$ is the cumulative displacement so that
$\boldsymbol{p}_{t_c}(u) = \boldsymbol{p}_0(u) + \boldsymbol{d}(u)$.
A vertex is \emph{displaced} whenever $\|\boldsymbol{d}(u)\| > 0$.

\subsubsection{Displacement-guided restoration}
Rather than tracking individual neighbor relationships, each agent
stores only its own original position and cumulative displacement vector.
The reformation objective for agent $u$ is to reduce
\[
r(u) := \|\boldsymbol{p}_t(u) - \boldsymbol{p}_0(u)\|_2
\]
to zero while preserving the connectivity restored during coagulation.

\subsubsection{Module selection}
At each reformation step, a single agent is selected to move. The
selection prioritizes agents that can reach their exact original position
in a single pivot, followed by agents with the largest displacement
magnitude:
\[
u^\star = \arg\max_{u \in \mathcal{C}_t}
\bigl(\mathbf{1}[\text{can\_reach\_original}(u)],\;
\|\boldsymbol{d}(u)\|\bigr),
\]
where $\mathcal{C}_t \subseteq \bar{V}_{t_c}$ is the set of displaced
agents that pass the mobility check (defined below), the tuple is
compared lexicographically, and ties are broken by agent identifier.

\subsubsection{Greedy pivot selection}
Once selected, the agent evaluates all feasible pivots and retains only those that \emph{strictly decrease} the
residual distance:
\[
\mathcal{P}_t(u) = \bigl\{\Delta\boldsymbol{p} :
\|\boldsymbol{p}_t(u) + \Delta\boldsymbol{p} - \boldsymbol{p}_0(u)\|_2
< r(u) \bigr\}.
\]
Among these, pivots that land exactly on $\boldsymbol{p}_0(u)$ are
preferred; otherwise the pivot with the largest distance reduction is
executed:
\begin{align*}
\Delta\boldsymbol{p}^\star = \arg\max_{\Delta\boldsymbol{p}\in\mathcal{P}_t(u)}
\bigl(&\mathbf{1}[\boldsymbol{p}_t(u){+}\Delta\boldsymbol{p}=\boldsymbol{p}_0(u)],\\
&r(u) - \|\boldsymbol{p}_t(u){+}\Delta\boldsymbol{p}-\boldsymbol{p}_0(u)\|_2\bigr).
\end{align*}

\subsubsection{Safe motion}
Reformation moves must not re-break connectivity once partial
connectivity has been restored. We therefore reuse the conservative
mobility test and local mutual exclusion defined in
Section~\ref{sec:crittest}--\ref{sec:pivotselection}: $u$ may move only
if it is a leaf vertex or is declared movable by the bounded-hop
criticality heuristic. Additionally, candidate destinations must not
coincide with any fault position or currently occupied lattice site. This
yields a sequential, greedy reformation process that reduces each
agent's displacement while preserving any newly formed connectivity.

\begin{algorithm}[t]
\caption{Restructuring Policy}
\label{alg:restructuring}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Movement histories $\{\boldsymbol{p}_0(u), \boldsymbol{d}(u)\}$, fault positions $\mathcal{F}$
\STATE \textbf{while} $\exists\, u$ with $r(u)>0$ \textbf{and} $\mathcal{C}_t \neq \emptyset$:
\STATE \hspace{1em} \textbf{for each} displaced $u$: compute $r(u) \gets \|\boldsymbol{p}_t(u)-\boldsymbol{p}_0(u)\|$
\STATE \hspace{1em} $\mathcal{C}_t \gets \{u : r(u)>0 \text{ and } u \text{ passes mobility check}\}$
\STATE \hspace{1em} \textbf{if} $\mathcal{C}_t = \emptyset$: \textbf{break}
\STATE \hspace{1em} $u^\star \gets \arg\max_{u\in\mathcal{C}_t}(\mathbf{1}[\text{can\_reach}],\; \|\boldsymbol{d}(u)\|)$
\STATE \hspace{1em} $\mathcal{P}_t(u^\star) \gets$ all pivots with $r'(u^\star)<r(u^\star)$, excluding $\mathcal{F}$ and occupied sites
\STATE \hspace{1em} \textbf{if} $\mathcal{P}_t(u^\star)=\emptyset$: \textbf{continue}
\STATE \hspace{1em} $\Delta\boldsymbol{p}^\star \gets \arg\max_{\Delta\boldsymbol{p}\in\mathcal{P}_t}(\mathbf{1}[\text{reaches original}],\; \text{distance reduction})$
\STATE \hspace{1em} Execute pivot $\Delta\boldsymbol{p}^\star$; update $\boldsymbol{d}(u^\star)$
\STATE \hspace{1em} Form new connections with adjacent modules
\end{algorithmic}
\end{algorithm}
\end{comment}

\subsection{Restructuring After Coagulation}
\label{sec:reformation}

After the coagulation phase, the network connectivity may be restored but
the original shape is lost. We therefore consider a two-phase strategy:
(i) \emph{coagulation}, in which agents migrate toward faults to restore
connectivity, and (ii) \emph{reformation}, in which agents partially
recover the pre-damage shape without re-breaking newly restored
connectivity. The restructuring approach is summarized in Algorithm \ref{alg:restructuring}.


Let $t_\mathrm{c}$ denote the time at which coagulation terminates and reformation
begins. We assume each surviving active vertex
$u \in \bar{V}_{t_\mathrm{c}}$ stores a finite set of \emph{memory edges} from a
reference time $t_0 \le t$ prior to fragmentation:
\[
N_0(u) \subseteq \bar{V}_{t_0},
\]
and the corresponding reference displacements
\[
g_{t_0}(u,v), \qquad v \in N_0(u),
\]
obtained from the pre-damage embedding.

\subsubsection{Reformation tokens}
During reformation, agents broadcast \emph{rendezvous tokens} requesting
reconnection to former neighbors. Each ego agent $u$ maintains a set of
tokens
\[
\mathcal{R}_t(u) \subseteq \bar{V}_{t_0} \times \mathbb{R}^3,
\]
where a token $(v,\boldsymbol{\zeta})$ indicates a request to reconnect to a specific
former neighbor $v$ along a direction vector $\boldsymbol{\zeta}$ expressed in the frame of $u$.

\paragraph{Token generation}
At each reformation update, $u$ generates a token for each $v\in
N_0(u)$ that is not currently adjacent to $u$:
\[
(v,\boldsymbol{\zeta}), \qquad \boldsymbol{\zeta} := \boldsymbol{\rho}_{uv}.
\]

\paragraph{Token propagation}
When $u$ sends a token $(v,\zeta)$ to neighbors $w\in\bar{N}_t(u)$, it updates it by
composition with the current edge translation:
\[
\boldsymbol{\zeta} \leftarrow \boldsymbol{\rho}_{wu}+ \boldsymbol{\zeta},
\]
and rebroadcasts $(v,\boldsymbol{\zeta})$.

\subsubsection{Furthest-neighbor selection}
To avoid local minima in which agents greedily reconnect to nearby
requests, the ego agent prioritizes \emph{furthest} targets. Specifically,
upon deciding to move, $u$ selects
\[
(v^\star,\boldsymbol{\zeta}^\star) = \argmax_{(v,\boldsymbol{\zeta})\in \mathcal{R}_t(u)} \|\boldsymbol{\zeta}\|_2,
\]
breaking ties arbitrarily or by recency.

\subsubsection{Safe motion and directed pivoting}
Reformation moves must not re-break connectivity once partial connectivity has been restored. We therefore reuse the conservative mobility test and local mutual exclusion defined in Section \ref{sec:crittest}--\ref{sec:pivotselection}: $u$ may move only if it is
declared movable by the bounded-hop criticality heuristic. The ego agent selects
\[
\Delta \boldsymbol{p}^\star = \arg\max_{\Delta \boldsymbol{p}} \langle \Delta \boldsymbol{p}, \boldsymbol{\xi}^\star \rangle,
\]
subject to the \emph{alignment constraint} $\langle \Delta \boldsymbol{p}^\star, \boldsymbol{\xi}^\star \rangle > 0$ and otherwise remains stationary. This yields an asynchronous,
stress-sharing-style reformation process that attempts to restore
adjacency relationships from $N_0(\cdot)$ while
preserving any newly formed connectivity.

\begin{algorithm}[t]
\caption{Restructuring Policy}
\label{alg:restructuring}
\begin{algorithmic}[1]
\STATE \textbf{def} restructuring$(u, \bar{N}_t(u), N_0(u), \mathcal{R}_t(u))$:
\STATE \hspace{1em} \textbf{while True}$:$
\STATE \hspace{2em} \textbf{for each} $v \in N_0(u)$:
\STATE \hspace{3em} $\boldsymbol{\zeta} \gets \boldsymbol{\rho}_{uv}$
\STATE \hspace{3em} $\mathcal{R}_{t+1}(u) \gets \mathcal{R}_t(u) \cup (v,\boldsymbol{\zeta})$
\STATE \hspace{2em} \textbf{for each} $(v,\boldsymbol{\zeta}) \in \mathcal{R}_t(u)$:
\STATE \hspace{3em} \textbf{for each} $w \in \bar{N}_t(u)$:
\STATE \hspace{4em} $\boldsymbol{\zeta} \gets \boldsymbol{\rho}_{wu}+\boldsymbol{\zeta}$
\STATE \hspace{4em} $\mathcal{R}_{t+1}(w) \gets \mathcal{R}_t(w) \cup (v,\boldsymbol{\zeta})$
\STATE \hspace{2em} \textbf{if} len$(\mathcal{R})=0$:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} movable $\gets$ Criticality Test (Section \ref{sec:crittest})
\STATE \hspace{2em} \textbf{if} \textbf{not} movable:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} $(v^\star,\boldsymbol{\zeta}^\star)\gets \arg\max_{(v,\boldsymbol{\zeta})\in \mathcal{R}_t(u)} \|\boldsymbol{\zeta}\|_2$
\STATE \hspace{2em} $\Delta \boldsymbol{p}^\star \gets \arg\max_{\Delta \boldsymbol{p}} \langle \Delta \boldsymbol{p}, \boldsymbol{\zeta}^\star \rangle$
\STATE \hspace{2em} \textbf{if} $\langle \Delta \boldsymbol{p}^\star, \boldsymbol{\zeta}^\star \rangle \le 0$:
\STATE \hspace{3em} \textbf{continue}
\STATE \hspace{2em} Execute $\Delta \boldsymbol{p}^\star$
\end{algorithmic}
\end{algorithm}

\begin{table}[t]
\centering
\caption{Monte Carlo Results Summary for 1000 Trials.}
\label{tab:results_summary_stacked}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.1}

\begin{tabular}{p{1.4cm} c|cccc}
\hline
\textbf{Metric} & $\boldsymbol{n}$ & \textbf{FC Single} & \textbf{FC Dynamic} & \textbf{Tree Single} & \textbf{Tree Dynamic} \\
\hline

\multirow{3}{=}{Reconnection\\Rate (\%)}
& 10  & 91.6 & 90.9 & 91.8 & 91.8 \\
& 50  & 100.0 & 98.0 & 100.0 & 99.2 \\
& 100 & 100.0 & 97.8 & 100.0 & 98.9 \\

\hline
\multirow{3}{=}{Total\\Difference}
& 10  & 0.165 & 0.164 & 0.110 & 0.109 \\
& 50  & 0.040 & 0.086 & 0.021 & 0.049 \\
& 100 & 0.023 & 0.071 & 0.011 & 0.040 \\

\hline
\multirow{3}{=}{Phase 1\\Moves}
& 10  & 3.7 & 3.7 & 2.7 & 2.7 \\
& 50  & 3.9 & 11.4 & 2.1 & 6.5 \\
& 100 & 4.7 & 19.7 & 2.1 & 11.2 \\

\hline
\multirow{3}{=}{Phase 2\\Moves}
& 10  & 0.6 & 0.6 & 0.7 & 0.7 \\
& 50  & 0.7 & 2.0 & 0.8 & 1.7 \\
& 100 & 0.8 & 3.3 & 0.7 & 2.4 \\

\hline
\multirow{3}{=}{Meaningful\\Trials}
& 10  & 426 & 426 & 499 & 499 \\
& 50  & 287 & 856 & 584 & 983 \\
& 100 & 251 & 951 & 580 & 1000 \\

\hline
\end{tabular}
\end{table}

\begin{figure}[!t]
  \centering
  \includegraphics[width=\linewidth]{overlay_reconnection_rate.png}
  \caption{Reconnection rate versus number of modules $n$ across Monte Carlo trials. Bold points denote average trial outcomes for a given number of modules (success/failure in restoring active connectivity) and curves show the smoothed trend for each configuration. Reconnection reliability increases with system size and approaches near-perfect rates for most configurations; dynamic fault scenarios remain slightly more challenging, while tree-based assemblies maintain the highest reconnection rates across $n$.}
  \label{fig:reconnection_rate_overlay}
\end{figure}

\begin{figure}[!t]
  \centering
  \includegraphics[width=\linewidth]{overlay_missing_portions.png}
  \caption{Missing-port fraction versus number of modules $n$ across Monte Carlo trials. Bold points show averages of randomized trials for a given number of modules, and solid/dashed curves show the smoothed trend for each configuration (FC Single, FC Dynamic, Tree Single, Tree Dynamic). Missing-port fraction decreases with system size for all configurations, with tree-based assemblies achieving consistently lower missing-port fractions than fully-connected assemblies, and dynamic faults incurring higher missing-port fractions than single-fault trials.}
  \label{fig:missing_portions_overlay}
\end{figure}

\section{Results}
\label{sec:results}

We evaluated the proposed coagulation–reformation policy via Monte Carlo simulations on assemblies with $n\in[10,100]$ modules. Initial structures were generated using two growth processes: a fully connected (FC) multi-neighbor lattice structure where each new module attaches to all adjacent existing neighbors, and a tree structure where each new module attaches to a single parent along a frontier. For each $n$, we tested (i) a single-fault scenario with one inactive module, and (ii) a dynamic-fault scenario with $n/10$ simultaneous failures.

Each trial executes coagulation for connectivity restoration followed by restructuring for shape recovery. We report: \emph{reconnection rate} (fraction of trials with a connected final graph), \emph{total difference} $\mathrm{diff}(P_t,P_{t_{\mathrm{f}}})$ (shape dissimilarity), \emph{Phase 1/2 moves} (mean pivot count), and \emph{meaningful trials} (post-damage disconnected cases). Table~\ref{tab:results_summary_stacked} summarizes key metrics for $n\in\{10,50,100\}$, and Figures~\ref{fig:reconnection_rate_overlay}--\ref{fig:missing_portions_overlay} show trends across all $n$.

\paragraph{Connectivity restoration}
Reconnection reliability increases with system size across all configurations. Tree structures achieve near-perfect recovery (≈100\% by $n=50$), while dense structures remain slightly worse under dynamic faults (≈98\% at $n=100$). Single faults are solved reliably in all cases.

\paragraph{Shape preservation}
Tree structures retain significantly more pre-damage adjacency. At higher $n$, shape loss is noticeably lower for trees than for dense structures across both fault models, indicating better preservation of local adjacency.

\paragraph{Repair effort}
Dynamic faults incur substantially more pivot motion in dense assemblies. Coagulation move counts grow with $n$ and are several times larger than in tree cases, while restructuring moves remain modest but consistently higher for dense graphs.

\paragraph{Effect of baseline connectivity}
Despite higher nominal redundancy, dense assemblies admit fewer locally admissible moves under connectivity-preserving constraints, occasionally producing local deadlocks and slightly lower reconnection rates. Trees contain more leaf-like modules that satisfy local mobility tests, increasing the set of feasible repair actions.

Overall, the stress-sharing policy reliably restores connectivity and partially recovers structure with performance improving with scale. Sparse structures, such as those with Vicsek-like connectivity~\cite{liao2025assembly}, benefit particularly under conservative local motion constraints.



\section{Conclusions}
In this work, we presented a decentralized stress-sharing reconfiguration strategy for modular space robots that restores connectivity after damage while reconstructing the structure to be similar to the pre-failure configuration. By modeling the system as a lattice-constrained graph and restricting actions to local connectivity-safe pivots, the method enables physically feasible repair using only local information. Future work will extend the framework to broader geometries and modular systems and develop theoretical guarantees, including reconnection convergence and bounds on achievable shape recovery.

These results suggest that resilience in space robotics may rely less on redundancy and more on distributed reorganization inspired by biological repair processes. Establishing mathematical and algorithmic principles for such homeostatic behavior could enable modular spacecraft to maintain function after unexpected failures and extend mission lifetime without external intervention.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
