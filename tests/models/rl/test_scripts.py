from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_double_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.double_dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_dueling_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.dueling_dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_n_step_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.n_step_dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_noisy_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.noisy_dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_per_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.per_dqn_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_reinforce(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.reinforce_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_rl_vanilla_policy_gradient(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.vanilla_policy_gradient_model import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()
