from typing import Any, Callable, Dict, List


class IReader:
    """Reader abstraction for all Readers.

    Applies a function to an element of your loader.
    For example to a row from csv, or to an image, etc.

    All inherited classes have to implement `__call__`.
    """

    def __init__(self, input_key: str, output_key: str) -> None:
        """Create a new instance of ReaderSpec.

        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Read a row from your annotations dict and transfer it to loader.

        Args:
            element: elem in your datasets

        Raises:
            NotImplementedError: Because this class is an abstraction of
        """
        raise NotImplementedError("You cannot apply a transformation using `BaseReader`")


class ReaderCompose:
    """Abstraction to compose several readers into one open function."""

    def __init__(self, readers: List[IReader]) -> None:
        """Create a new instance of ReaderCompose.

        Args:
            readers: list of readers to compose
        """
        self.readers = readers

    def __call__(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Read a row from your annotations dict nd applies all readers and mixins.

        Args:
            element: elem in your datasets.

        Returns:
            Value after applying all readers and mixins
        """
        result: Dict[str, Any] = dict()
        for fn in self.readers:
            result = {**result, **fn(element)}
        return result


class Augmentor:
    """Augmentation abstraction to use with loader dictionaries."""

    def __init__(
        self,
        input_key: str,
        output_key: str,
        augment_fn: Callable,  # type: ignore[type-arg]
    ) -> None:
        """Create a new instance of Augmentor.

        Args:
            augment_fn: augmentation function to use
            input_key: ``augment_fn`` input key
            output_key: ``augment_fn`` output key
        """
        self.input_key = input_key
        self.output_key = output_key if output_key is not None else input_key
        self.augment_fn = augment_fn

    def __call__(self, dict_: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the augmentation."""
        values = dict_[self.input_key]
        dict_[self.output_key] = self.augment_fn(values)
        return dict_


class AugmentorCompose:
    """Compose augmentors."""

    def __init__(self, key2augment_fn: Dict[str, Callable]) -> None:  # type: ignore[type-arg]
        """Create a new instance of AugmentorCompose.

        Args:
            key2augment_fn: mapping from input key to augmentation function to apply
        """
        self.key2augment_fn = key2augment_fn

    def __call__(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the augmentation.

        Args:
            dictionary: item from datasets

        Returns:
            dict: dictionaty with augmented loader
        """
        results: Dict[str, Any] = dict()
        for key, augment_fn in self.key2augment_fn.items():
            results = {**results, **augment_fn({key: dictionary[key]})}

        return {**dictionary, **results}


__all__ = ["Augmentor", "AugmentorCompose", "IReader", "ReaderCompose"]
