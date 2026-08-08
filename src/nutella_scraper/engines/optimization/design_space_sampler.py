"""Design space sampling and validation."""

from __future__ import annotations

from nutella_scraper.domain.models.design_space import DesignSpace, ParameterSample


class DesignSpaceSampler:
    """Samples parameter vectors from DesignSpace."""

    def sample(self, design_space: DesignSpace) -> ParameterSample:
        raise NotImplementedError("DesignSpaceSampler.sample not implemented")

    def validate(self, design_space: DesignSpace, sample: ParameterSample) -> bool:
        raise NotImplementedError("DesignSpaceSampler.validate not implemented")
