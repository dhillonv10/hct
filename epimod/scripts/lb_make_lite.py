"""One-off: build the compact CpG universe and convert the ComputAge store to it."""
from epimod.lb.lite import convert_store
from epimod.lb.tasks import load_parsed

convert_store(load_parsed())
