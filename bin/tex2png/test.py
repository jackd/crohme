from subprocess import call

call(['./tex2png', '-b', 'rgb 1 1 1', '-c', '$ax^2 + bx + c < 0$', '--',
      '-T', 'tight'])
