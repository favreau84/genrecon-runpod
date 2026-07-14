# Stub minimal de mmcv pour NonCuboidRoom (Plane-DUSt3R).
# mmcv==1.2.1 (pin upstream) ne compile pas sur Python 3.11 ; seul
# noncuboid/models/hrnet.py l'importe, et n'utilise que constant_init,
# kaiming_init (init de poids) et load_checkpoint (jamais appelé à
# l'inférence : Detector fait init_weights(pretrained=None)).
__version__ = "1.2.1-stub"
