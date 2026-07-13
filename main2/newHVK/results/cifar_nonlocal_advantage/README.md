# CIFAR nonlocal advantage diagnostic

This suite uses real CIFAR-10 images but changes the task from ordinary image
reconstruction to nonlocal patch-correlation prediction. The target depends on
products between distant patch statistics, so an entangling pair-observable map
has an explicit representational route that single-site/local controls lack.
Because this target is constructed from pair products, the suite also includes
explicit quadratic-pair and degree-two polynomial-feature classical controls.
If those controls match the entangling map, the result should be interpreted as
a pair-feature inductive-bias diagnostic rather than a uniquely quantum effect.

Claim boundary: this is a CIFAR-derived entanglement-sensitive representational
advantage diagnostic. It is not ordinary CIFAR reconstruction advantage and not
a hardware quantum-advantage proof.
