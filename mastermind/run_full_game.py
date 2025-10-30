import json
from argparse import ArgumentParser

from mastermind.evaluator import Evaluator
from mastermind.game import Mastermind
from mastermind.models import AnthropicModel, HFModel, OpenAIModel
from mastermind.solvers import KnuthSolver
from mastermind.utils import print_summary

# === NEW: 引入 Letta 适配器（你已放在 mastermind/src/mastermind/letta_model.py） ===
from mastermind.letta_model import LettaModel
from mastermind.letta_auto_memory_model import LettaAutoMemoryModel

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--model_type", type=str, default="hf", help="Model type.")
    parser.add_argument("--generation_args", type=str, default={}, help="Generation kwargs.")
    parser.add_argument("--use_cot", action="store_true", help="Use COT.")
    parser.add_argument("--use_full_example", action="store_true", help="Use full example.")
    parser.add_argument("--code_length", type=int, default=4, help="Code length of the game.")
    parser.add_argument("--num_colors", type=int, default=6, help="Number of colors in the game.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--save_results", action="store_true", help="Save results.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results.")

    # === NEW: Letta 服务相关参数（若启用 SECURE=true，可用 token 传入密码） ===
    parser.add_argument("--letta_url", type=str, default="http://localhost:8283", help="Letta server base URL")
    parser.add_argument("--letta_token", type=str, default=None, help="Password token if SECURE=true")
    # ✅ 新增参数：控制 Letta 工具行为
    parser.add_argument("--letta_tools_hint", action="store_true", help="Enable minimal tool hint")
    parser.add_argument("--letta_force_tools", action="store_true", help="Force using memory tools each turn")

    args = parser.parse_args()

    if args.generation_args:
        generation_args = json.loads(args.generation_args)  # Convert JSON string to a dictionary
    else:
        generation_args = None

    game = Mastermind(code_length=args.code_length, num_colors=args.num_colors)
    # ✅ 开局就打印这一局的目标答案
    print(f"\n🎯 本局目标答案（secret_code）: {game.secret_code}\n")

    if args.model_type == "hf":
        model = HFModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "openai":
        model = OpenAIModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "anthropic":
        model = AnthropicModel(model_name=args.model, generation_args=generation_args)
    elif args.model_type == "letta":
        # Letta 模型命名通常为 "openai/<name>"；如果你传 "gpt-4o-mini" 我们自动补前缀
        backend_model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"
        model = LettaModel(
            base_url=args.letta_url,
            model=backend_model,
            token=args.letta_token,
            verbose=True,
        )
    elif args.model_type == "letta_auto":
        backend_model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"
        model = LettaAutoMemoryModel(
            base_url=args.letta_url,  # or 使用 token=... 直连托管
            model=backend_model,  # 与 lette 分支一致：自动补上 openai/ 前缀
            embedding="openai/text-embedding-3-small",  # Letta 侧用来做记忆检索的向量配置
            token=args.letta_token,
            verbose=True,
            print_mem_each_round=True,
            tools_mode="auto",
            tools_hint=args.letta_tools_hint,  # ✅
            force_tools=args.letta_force_tools,
        )

    elif args.model_type == "knuth":
        model = KnuthSolver(game)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    evaluator = Evaluator(game, model, use_cot=args.use_cot, use_fewshot_example=args.use_full_example)
    result = evaluator.run(
        num_games=args.num_runs, save_results=args.save_results, save_path=args.save_path, compute_progress=True
    )
    print_summary(model, game, result, args.num_runs)
