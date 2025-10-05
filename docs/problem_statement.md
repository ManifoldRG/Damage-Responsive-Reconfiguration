\documentclass{article}
\usepackage{graphicx}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amsthm}

\newtheorem{lemma}{Lemma}

\title{Damage-Responsive Reconfiguration}

\begin{document}

\maketitle

\section{Problem Formulation}

We represent the modular spacecraft at time $t$ as a
\emph{Unit Dual Quaternion Directed Graph} (UDQDG)
\[
\mathcal{G}_t = (V_t, E_t, m_t, Q_t, \hat q_t),
\]
with:
\begin{itemize}
    \item $V_t$ the set of modules (vertices).
    \item $E_t \subset V_t \times V_t$ the set of directed edges representing physical connections.
    \item $m_t: V_t \to \{0,1\}$ the \emph{activity map} (active $=1$, inactive $=0$).
    \item $Q_t: V_t \to \mathbb{S}^3$ (optional) the \emph{absolute attitude} (unit quaternion) per vertex; $Q_t$ is an independent attribute and does not affect geometry below.
    \item $\hat q_t : E_t \to \mathbb{DH}_u$ the \emph{edge gains}, restricted to
          \emph{pure translations}:
          \[
             \hat q_{uv} \;=\; 1 + \frac{\varepsilon}{2}\,q^{\mathrm{tr}}_{uv}, \qquad
             q^{\mathrm{tr}}_{uv} = (0,\; \mathbf t_{uv}), \ \ \mathbf t_{uv}\in\mathbb{R}^3,
          \]
          where $\varepsilon^2=0$ and $q^{\mathrm{tr}}_{uv}$ is a pure imaginary quaternion encoding the translation vector $\mathbf t_{uv}$. Gains satisfy
          \[
             \hat q_{vu} \;=\; \hat q_{uv}^{-1} \;=\; 1 - \frac{\varepsilon}{2}\,q^{\mathrm{tr}}_{uv},
          \]
          i.e., $\mathbf t_{vu} = -\,\mathbf t_{uv}$.
\end{itemize}

\paragraph{Active connectivity.}
The \emph{active subgraph} is
\[
\mathcal{G}^A_t = (V^A_t, E^A_t),\qquad
V^A_t = \{v \in V_t \mid m_t(v)=1\},\quad
E^A_t = \{(u,v)\in E_t \mid m_t(u)=m_t(v)=1\}.
\]
Only edges in $E^A_t$ participate in reconfiguration.

\paragraph{Geometric consistency (translation-only).}
Along any active path
$P : r=v_0 \to v_1 \to \cdots \to v_k=v$,
the cumulative transform is
\[
\hat q_{rv} \;=\; \prod_{i=0}^{k-1} \left(1 + \frac{\varepsilon}{2} q^{\mathrm{tr}}_{v_i v_{i+1}}\right)
\;=\; 1 + \frac{\varepsilon}{2}\sum_{i=0}^{k-1} q^{\mathrm{tr}}_{v_i v_{i+1}},
\]
since $\varepsilon^2=0$. Equivalently, positions (up to a global translation) are obtained by summing translations:
\[
\mathbf p_v \;=\; \sum_{i=0}^{k-1} \mathbf t_{v_i v_{i+1}} \in \mathbb{R}^3.
\]
Path-independence holds iff every directed cycle $C$ in $\mathcal{G}^A_t$ satisfies the closure condition
\[
\sum_{(i\to j)\in C} \mathbf t_{ij} \;=\; \mathbf 0
\quad\Longleftrightarrow\quad
\prod_{(i\to j)\in C} \hat q_{ij} \;=\; 1.
\]

\paragraph{Lattice constraint.}
To enforce a lattice, constrain each translation to be a unit step in a discrete direction set:
\[
\mathcal{D} \subset \mathbb{S}^2, \qquad
\|\mathbf t_{uv}\| = 1, \ \ \frac{\mathbf t_{uv}}{\|\mathbf t_{uv}\|} \in \mathcal{D}, \ \ \mathbf t_{vu}=-\mathbf t_{uv}.
\]
For a cubic lattice, $\mathcal{D} = \{\pm \hat{\mathbf x},\,\pm \hat{\mathbf y},\,\pm \hat{\mathbf z}\}$.

\subsection*{Damage-Responsive Reconfiguration Objective}
A \emph{damage event} at time $t$ changes activity of one or more vertices:
\[
m_{t+1}(v) = 0 \quad \text{for some } v \text{ with } m_t(v)=1.
\]
Inactive vertices remain physically connected but cannot pivot or serve as pivot axes.

The goal is to find a finite sequence of allowable \emph{pivot operations} that restores connectivity of the active graph in the fewest steps.

\subsection*{Pivotability Rules}
A vertex $u\in V_t$:
\begin{itemize}
    \item \textbf{Can pivot} if $m_t(u)=1$ and all neighbors $v$ with $(u,v)\in E_t$ are also active.
    \item \textbf{Can be pivoted on} if $m_t(u)=1$.
\end{itemize}
Inactive vertices cannot serve as pivot axes.

\subsection*{Pivot Operations in Dual-Quaternion (Translation) Space}
A \emph{pivot} updates translations without using lattice coordinates:
\begin{enumerate}
    \item Choose an active pivot axis $p$ and an active neighbor $u$ with $(p,u)\in E^A_t$.
    \item Detach $u$ from $p$ (remove $(p,u)$ and $(u,p)$) and reattach at a new unit direction $\tilde{\mathbf t}\in\mathcal D$ by inserting edges with gains
    \[
        \hat q'_{pu} = 1 + \frac{\varepsilon}{2}(0,\tilde{\mathbf t}), \qquad
        \hat q'_{up} = \left(\hat q'_{pu}\right)^{-1} = 1 - \frac{\varepsilon}{2}(0,\tilde{\mathbf t}).
    \]
    \item Treat the connected component of $u$ (excluding $p$) as a rigid substructure in the \emph{relative} sense: internal edges keep their translations; only the cut edge at $p$ is changed.
\end{enumerate}
The transition set $T$ consists of all such updates that maintain validity.

\subsection*{Optimization Problem}

Given $\mathcal{G}_t$ at time $t$ and a damage event producing the activity update $m_{t+1}$, find an integer
\[
t_f \ge t+1
\]
and a sequence
\[
\{\mathcal{G}_{t+1}, \mathcal{G}_{t+2}, \dots, \mathcal{G}_{t_f}\}
\]
such that:
\begin{align*}
\min_{\{\mathcal{G}_{\tau}\}_{\tau=t+1}^{t_f}} \quad & t_f \\
\text{s.t.}\quad
& (\mathcal{G}_{\tau}, \mathcal{G}_{\tau+1}) \in T 
&& \forall \tau \in \{t+1,\dots,t_f-1\} \quad \text{(valid pivot transitions)}, \\
& \mathcal{G}_{\tau} \in S 
&& \forall \tau \in \{t+1,\dots,t_f\} \quad \text{(all intermediate states allowable)}, \\
& \mathcal{G}^A_{t_f} \ \text{is connected} 
&& \text{(final active graph connected)}, \\
& \mathcal{G}_{t+1} \ \text{matches the damage at time } t 
&& \text{(boundary condition)}.
\end{align*}

\noindent
Here, $S$ is the set of structurally valid UDQDG states under the translation-only restriction (unit dual quaternion edges with identity rotation; unit-length lattice-constrained translations; and cycle-closure where required), and $T$ is the set of allowable pivot transitions that respect pivotability and preserve validity.

\subsection*{Graph-Theoretic Pivoting Rules (In Progress)}
Each directed edge $(u,v)\in E_t$ carries a unit dual quaternion $\hat q_{uv}$ representing a pure translation, with 
\[
d = \mathrm{trans}(\hat q_{uv}) \in \mathcal{D}, \qquad 
\hat q_{vu} = \hat q_{uv}^{-1}, \quad \mathrm{trans}(\hat q_{vu})=-d.
\]
Here $\mathcal{D}=\{\pm \hat{\mathbf{x}}, \pm \hat{\mathbf{y}}, \pm \hat{\mathbf{z}}\}$ is the set of allowable lattice directions.  
Each vertex face may be occupied by at most one edge (\emph{port exclusivity}).  
A vertex $u$ is \emph{pivotable} if $m(u)=1$ and all its neighbors are active.

\vspace{1ex}
\noindent\textbf{Corner Pivot.}  
A \emph{corner pivot} moves a module $u$ around a corner of its neighbor $v$, changing which face of $v$ it attaches to:
\begin{itemize}
    \item Current edge: $(u,v)$ with $d=\mathrm{trans}(\hat q_{uv})$.
    \item Choose a new direction $d'\in\mathcal{D}$ such that:
    \begin{enumerate}
        \item $d\cdot d'=0$ (orthogonal directions).
        \item $v$ has no other neighbor using face $-d'$.
        \item $u$ has no other neighbor using face $d'$.
        \item $m(u)=m(v)=1$ and all neighbors of $u$ are active.
    \end{enumerate}
    \item Update the edge:
    \[
    \hat q_{uv} \leftarrow \text{UDQ}(d'), \qquad
    \hat q_{vu} \leftarrow \hat q_{uv}^{-1}.
    \]
\end{itemize}

\vspace{1ex}
\noindent\textbf{Flat Pivot.}  
A \emph{flat pivot} rolls a module $u$ off of its neighbor $v$ and onto a different neighbor $w$, while keeping the same face direction $d$:
\begin{itemize}
    \item Current edge: $(u,v)$ with $d=\mathrm{trans}(\hat q_{uv})$.
    \item Select another neighbor $w$ of $v$ such that:
    \begin{enumerate}
        \item $w$ has no other neighbor using face $-d$.
        \item $u$ has no other neighbor using face $d$.
        \item $m(u)=m(v)=m(w)=1$ and all neighbors of $u$ are active.
    \end{enumerate}
    \item Update the connection:
    \[
    \text{remove }(u,v),\quad
    \text{add }(u,w)\text{ with }\mathrm{trans}(\hat q'_{uw})=d,\quad
    \hat q'_{wu} = (\hat q'_{uw})^{-1}.
    \]
\end{itemize}

\noindent
In both pivot types, only the specified edge(s) are updated; all other edges remain unchanged. These operations preserve the lattice constraints and respect the port exclusivity rules.

\subsection*{Connectivity Recovery Lemma}
\begin{lemma}[Connectivity-Preserving Reconfiguration]
Let $\mathcal{G}^A_t$ be disconnected immediately after a damage event at time $t$. Suppose there exist components $C_1,C_2\subset V^A_t$ and a vertex $u\in C_1$ such that:
\begin{itemize}
    \item $u$ is pivotable,
    \item there exists a pivot operation assigning a new pure-translation gain $\hat q'_{pu}$ whose translation is $\tilde{\mathbf t}\in\mathcal D$ so that $u$ becomes adjacent (unit step) to some $v\in C_2$ with $m_t(v)=1$,
\end{itemize}
then there exists a valid transition $\mathcal{G}_{t+1}$ such that $\mathcal{G}^A_{t+1}$ is connected.
\end{lemma}

\end{document}
