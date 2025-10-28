"""
Validator routing system for quality assessment.

Routes quality checks to appropriate tables based on their role (base vs embedding)
and schema. Prevents inappropriate checks (e.g., text checks on embedding tables).
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import yaml

from lakehouse.logger import get_default_logger


logger = get_default_logger()


class ValidatorRouter:
    """
    Routes quality checks to appropriate tables based on configuration.
    
    Loads validator_routing.yaml and provides methods to:
    - Determine if a check should run on a table
    - Get table role (base vs embedding)
    - Validate routing configuration
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize validator router.
        
        Args:
            config_path: Path to validator_routing.yaml (default: config/validator_routing.yaml)
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path is None:
            config_path = Path("config") / "validator_routing.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Validator routing configuration file not found: {config_path}\n"
                f"Please create {config_path} with table role definitions."
            )
        
        logger.info(f"Loading validator routing configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Validator routing configuration file is empty: {config_path}")
        
        # Validate and store configuration
        self._validate_config(config, config_path)
        
        self.tables: Dict[str, Dict[str, Any]] = config.get("tables", {})
        self.check_requirements: Dict[str, Dict[str, Any]] = config.get("check_requirements", {})
        self.error_handling: Dict[str, str] = config.get("error_handling", {})
        
        # Build lookup indices for fast access
        self._table_roles: Dict[str, str] = {}
        self._table_checks: Dict[str, Set[str]] = {}
        
        for table_name, table_config in self.tables.items():
            self._table_roles[table_name] = table_config.get("role", "unknown")
            self._table_checks[table_name] = set(table_config.get("checks", []))
        
        logger.info(
            f"Loaded validator routing: {len(self.tables)} tables, "
            f"{len(self.check_requirements)} check types"
        )
    
    def _validate_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Loaded configuration dictionary
            config_path: Path to config file (for error messages)
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level keys
        if "tables" not in config:
            raise ValueError(
                f"Validator routing configuration missing 'tables' section\n"
                f"File: {config_path}"
            )
        
        tables = config.get("tables", {})
        if not isinstance(tables, dict):
            raise ValueError(
                f"'tables' must be a dictionary\n"
                f"File: {config_path}"
            )
        
        # Validate each table configuration
        valid_roles = {"base", "embedding"}
        for table_name, table_config in tables.items():
            if not isinstance(table_config, dict):
                raise ValueError(
                    f"Table '{table_name}' configuration must be a dictionary\n"
                    f"File: {config_path}"
                )
            
            # Check role is valid
            role = table_config.get("role")
            if role not in valid_roles:
                raise ValueError(
                    f"Table '{table_name}' has invalid role '{role}'\n"
                    f"Valid roles: {valid_roles}\n"
                    f"File: {config_path}"
                )
            
            # Check checks is a list
            checks = table_config.get("checks", [])
            if not isinstance(checks, list):
                raise ValueError(
                    f"Table '{table_name}' checks must be a list\n"
                    f"File: {config_path}"
                )
    
    def should_run_check(self, table_name: str, check_name: str) -> bool:
        """
        Determine if a check should run on a table.
        
        Args:
            table_name: Name of the table (e.g., "spans", "span_embeddings")
            check_name: Name of the check (e.g., "coverage", "dim_consistency")
        
        Returns:
            True if check should run, False otherwise
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.should_run_check("spans", "coverage")
            True
            >>> router.should_run_check("span_embeddings", "coverage")
            False
        """
        # If table not in config, log warning and allow check
        if table_name not in self._table_checks:
            if self.error_handling.get("log_skipped_checks", True):
                logger.warning(
                    f"Table '{table_name}' not in validator routing config. "
                    f"Allowing check '{check_name}' by default."
                )
            return True
        
        # Check if this check is configured for this table
        is_configured = check_name in self._table_checks[table_name]
        
        if not is_configured and self.error_handling.get("log_skipped_checks", True):
            logger.debug(
                f"Skipping check '{check_name}' for table '{table_name}' "
                f"(not applicable for {self.get_table_role(table_name)} tables)"
            )
        
        return is_configured
    
    def get_table_role(self, table_name: str) -> str:
        """
        Get the role of a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            Role string: "base", "embedding", or "unknown"
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.get_table_role("spans")
            'base'
            >>> router.get_table_role("span_embeddings")
            'embedding'
        """
        return self._table_roles.get(table_name, "unknown")
    
    def get_checks_for_table(self, table_name: str) -> List[str]:
        """
        Get all checks configured for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            List of check names
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.get_checks_for_table("spans")
            ['coverage', 'length_buckets', 'duplicates', ...]
        """
        return list(self._table_checks.get(table_name, set()))
    
    def get_tables_by_role(self, role: str) -> List[str]:
        """
        Get all tables with a specific role.
        
        Args:
            role: Role to filter by ("base" or "embedding")
        
        Returns:
            List of table names
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.get_tables_by_role("base")
            ['spans', 'beats', 'sections']
        """
        return [
            table_name
            for table_name, table_role in self._table_roles.items()
            if table_role == role
        ]
    
    def get_check_requirements(self, check_name: str) -> Optional[Dict[str, Any]]:
        """
        Get requirements for a specific check.
        
        Args:
            check_name: Name of the check
        
        Returns:
            Dictionary with check requirements or None if not defined
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.get_check_requirements("coverage")
            {'required_columns': ['start_time', 'end_time', 'episode_id'], ...}
        """
        return self.check_requirements.get(check_name)
    
    def validate_check_for_table(
        self,
        table_name: str,
        check_name: str,
        available_columns: Set[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that a check can run on a table given its columns.
        
        Args:
            table_name: Name of the table
            check_name: Name of the check
            available_columns: Set of column names in the table
        
        Returns:
            Tuple of (can_run: bool, reason: Optional[str])
            If can_run is False, reason explains why
        
        Example:
            >>> router = ValidatorRouter()
            >>> router.validate_check_for_table("spans", "coverage", {"start_time", "end_time"})
            (False, "Missing required columns: {'episode_id'}")
        """
        # First check if check is configured for this table
        if not self.should_run_check(table_name, check_name):
            return False, f"Check '{check_name}' not configured for table '{table_name}'"
        
        # Get check requirements
        requirements = self.get_check_requirements(check_name)
        if not requirements:
            # No requirements defined, assume it can run
            return True, None
        
        # Check required columns
        required_columns = set(requirements.get("required_columns", []))
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        return True, None
    
    def generate_routing_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the routing configuration.
        
        Returns:
            Dictionary with routing summary statistics
        
        Example:
            >>> router = ValidatorRouter()
            >>> summary = router.generate_routing_summary()
            >>> summary['base_tables']
            ['spans', 'beats', 'sections']
        """
        base_tables = self.get_tables_by_role("base")
        embedding_tables = self.get_tables_by_role("embedding")
        
        # Count checks per table
        checks_per_table = {
            table_name: len(checks)
            for table_name, checks in self._table_checks.items()
        }
        
        return {
            "total_tables": len(self.tables),
            "base_tables": base_tables,
            "embedding_tables": embedding_tables,
            "base_table_count": len(base_tables),
            "embedding_table_count": len(embedding_tables),
            "total_check_types": len(self.check_requirements),
            "checks_per_table": checks_per_table,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of routing configuration
        """
        return {
            "tables": self.tables,
            "check_requirements": self.check_requirements,
            "error_handling": self.error_handling,
            "summary": self.generate_routing_summary(),
        }

