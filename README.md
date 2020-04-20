# readme2tex
Renders LaTeX for Github Readmes

<p align="center"><img alt="$$&#10;\huge\text{Hello \LaTeX}&#10;$$" src="svgs/d27ecd9d6334c7a020001926c8000801.png?invert_in_darkmode" align=middle width="159.690135pt" height="30.925785pt"/></p>

<p align="center"><img alt="\begin{tikzpicture}&#10;\newcounter{density}&#10;\setcounter{density}{20}&#10;    \def\couleur{blue}&#10;    \path[coordinate] (0,0)  coordinate(A)&#10;                ++( 60:6cm) coordinate(B)&#10;                ++(-60:6cm) coordinate(C);&#10;    \draw[fill=\couleur!\thedensity] (A) -- (B) -- (C) -- cycle;&#10;    \foreach \x in {1,...,15}{%&#10;        \pgfmathsetcounter{density}{\thedensity+10}&#10;        \setcounter{density}{\thedensity}&#10;        \path[coordinate] coordinate(X) at (A){};&#10;        \path[coordinate] (A) -- (B) coordinate[pos=.15](A)&#10;                            -- (C) coordinate[pos=.15](B)&#10;                            -- (X) coordinate[pos=.15](C);&#10;        \draw[fill=\couleur!\thedensity] (A)--(B)--(C)--cycle;&#10;    }&#10;\end{tikzpicture}" src="svgs/a00f34be6b1ce8e4820c9852c5e6163e.png" align=middle width="281.2887pt" height="243.69345pt"/></p>

<sub>**Make sure that pdflatex is installed on your system.**</sub>

----------------------------------------

`readme2tex` is a Python script that "texifies" your readme. It takes in Github Markdown and
replaces anything enclosed between dollar signs with rendered <img alt="$\text{\LaTeX}$" src="svgs/c068b57af6b6fa949824f73dcb828783.png?invert_in_darkmode" align=middle width="42.05817pt" height="22.407pt"/>.

In addition, while other Github TeX renderers tend to give a jumpy look to the compiled text, 
<p align="center">
<img src="http://i.imgur.com/XSV1rPw.png?1" width=500/>
</p>

`readme2tex` ensures that inline mathematical expressions
are properly aligned with the rest of the text to give a more natural look to the document. For example,
this formula <img alt="$\frac{dy}{dx}$" src="svgs/24a7d013bfb0af0838f476055fc6e1ef.png?invert_in_darkmode" align=middle width="14.297415pt" height="30.58869pt"/> is preprocessed so that it lines up at the correct baseline for the text.
This is the one salient feature of this package compared to the others out there.

### Installation

Make sure that you have Python 2.7 or above and `pip` installed. In addition, you'll need to have the programs `latex` 
and `dvisvgm` on your `PATH`. In addition, you'll need to pre-install the `geometry` package in <img alt="$\text{\LaTeX}$" src="svgs/c068b57af6b6fa949824f73dcb828783.png?invert_in_darkmode" align=middle width="42.05817pt" height="22.407pt"/>.

To install `readme2tex`, you'll need to run

```bash
sudo pip install readme2tex
```

or, if you want to try out the bleeding edge,

```bash
git clone https://github.com/leegao/readme2tex
cd readme2tex
python setup.py develop
```

To compile `INPUT.md` and render all of its formulas, run

```bash
python -m readme2tex --output README.md INPUT.md
```

If you want to do this automatically for every commit of INPUT.md, you can use the `--add-git-hook` command once to
set up the post-commit hook, like so

```bash
git stash --include-untracked
git branch svgs # if this isn't already there

python -m readme2tex --output README.md --branch svgs --usepackage tikz INPUT.md --add-git-hook

# modify INPUT.md

git add INPUT.md
git commit -a -m "updated readme"

git stash pop
```

and every `git commit` that touches `INPUT.md` from now on will allow you to automatically run `readme2tex` on it, saving
you from having to remember how `readme2tex` works. The caveat is that if you use a GUI to interact with git, things
might get a bit wonky. In particular, `readme2tex` will just assume that you're fine with all of the changes and won't
prompt you for verification like it does on the terminal.

<p align="center">
<a href="https://asciinema.org/a/2am62r2x2udg1zqyb6r3kpm1i"><img src="https://asciinema.org/a/2am62r2x2udg1zqyb6r3kpm1i.png" width=600/></a>
</p>

You can uninstall the hook by deleting `.git/hooks/post-commit`. See `python -m readme2tex --help` for a list
of what you can do in `readme2tex`.

### Examples:

Here's a display level formula
<p align="center"><img alt="$$&#10;\frac{n!}{k!(n-k)!} = {n \choose k}&#10;$$" src="svgs/32737e0a8d5a4cf32ba3ab1b74902ab7.png?invert_in_darkmode" align=middle width="127.89183pt" height="39.30498pt"/></p>

The code that was used to render this formula is just

    $$
    \frac{n!}{k!(n-k)!} = {n \choose k}
    $$

<sub>*Note: you can escape \$ so that they don't render.*</sub>

Here's an inline formula.











# Optimal Estimation of Temperature Change

```python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt = 0.0005
dx = 0.0005
k = 10**(-4)
x_max = 0.04
t_max = 1
T0 = 100

def FTCS(dt,dx,t_max,x_max,k,T0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    T = np.zeros([r,c])
    T[0,:] = T0*np.sin(np.pi/x_max*x)
    for n in range(0,r-1):
        for j in range(1,c-1):
            T[n+1,j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j+1]) 
        j = c-1 
        T[n+1, j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j-1])
        j = 0
        T[n+1, j] = T[n,j] + s*(T[n,j+1] - 2*T[n,j] + T[n,j+1])
    return x,T,r,s
    

x,T,r,s = FTCS(dt,dx,t_max,x_max,k,T0)

#plot_times = np.arange(0.01,1.0,0.01)
#for t in plot_times:
#    plt.plot(y,T[int(t/dt),:])


fig = plt.figure()
ax = plt.axes(xlim=(0, 0.04), ylim=(0,100))
line, = ax.plot([], [], lw=2)

plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Change in Temperature')

def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, T[i,:])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=2001, interval=1, blit=True)

```
