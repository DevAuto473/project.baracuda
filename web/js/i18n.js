// English / Arabic strings and language switching (Arabic switches the page to right-to-left).
const STRINGS = {
  en: {
    title: 'Barracuda - play the engine',
    tag: 'C++20 chess engine · self-trained NNUE · runs in your browser',
    eloTitle: 'Estimated strength of the published network',
    evalTitle: "Evaluation (White's point of view)",
    boardLabel: 'Chess board',
    game: 'Game', level: 'Strength', youPlay: 'You play', white: 'White', black: 'Black', random: 'Random',
    newGame: 'New game', undo: 'Undo', flip: 'Flip',
    engine: 'Engine', thinking: 'Thinking', depth: 'Depth', speed: 'Speed',
    network: 'Network', evaluation: 'Evaluation', trained: 'Trained', netStrength: 'Strength', updated: 'Updated',
    moves: 'Moves',
    footer: 'Runs entirely in your browser with WebAssembly — no server, no account. Engine source on GitHub.',
    idle: 'idle', thinkingNow: 'thinking…',
    loading: 'Loading engine…', engineThinking: 'Barracuda is thinking…',
    yourMove: 'Your move', inCheck: ' — you are in check',
    youWin: ' — you win!', engineWins: ' — Barracuda wins',
    engineError: 'Engine error: ', startError: 'Could not start the engine: ',
    netNnue: 'NNUE (neural network)', netClassic: 'Classical (hand-crafted)', notYet: 'not yet',
    trainedFmt: '{epochs} epochs, {m}M samples', eloFmt: '~{elo} Elo',
    promote: 'Promote to',
    lvl1: 'Beginner', lvl2: 'Casual', lvl3: 'Club player', lvl4: 'Advanced', lvl5: 'Expert', lvl6: 'Maximum',
    'reason.checkmate': 'checkmate', 'reason.stalemate': 'stalemate', 'reason.50-move rule': '50-move rule',
    'reason.insufficient material': 'insufficient material', 'reason.threefold repetition': 'threefold repetition',
    langButton: 'العربية',
    'viz.title': 'Neural network',
    'viz.thinking': 'Thinking: the network is evaluating positions from the line being analysed',
    'viz.idle': 'Network view of the current position',
    'viz.used': 'This network is playing for Barracuda.',
    'viz.notUsed': 'Visualisation only: Barracuda still plays with its classical evaluator until this network beats it in testing.',
    'viz.none': 'No network has been published yet, so this is only an illustration of the architecture.',
    'viz.l0': 'Input · pieces', 'viz.l1': 'Feature transformer', 'viz.l2': 'Hidden 1', 'viz.l3': 'Hidden 2', 'viz.l4': 'Output',
    'viz.sub0': '{n} active of 768', 'viz.sub1': '32 of 512 shown', 'viz.sub2': '32 neurons', 'viz.sub4': 'eval (White)',
    netClassicViz: 'Classical · network in training',
    'viz.s0': 'Input', 'viz.s1': 'Features', 'viz.s2': 'Hidden 1', 'viz.s3': 'Hidden 2', 'viz.s4': 'Output',
    'tab.game': 'Game', 'tab.net': 'Network', 'tab.moves': 'Moves',
    'thanks.title': 'Thank you!',
    'thanks.msg': 'Barracuda has been taught from the way you think, and your mind has become part of its neural network.',
    'thanks.plain': 'Thanks for playing!',
    'thanks.sent': 'Your game was saved anonymously (moves and result only) and will be used to train Barracuda.',
    'thanks.off': 'Sharing is switched off, so this game was not saved.',
    'thanks.show': 'Show the board',
    'share.label': 'Share my games anonymously so Barracuda can learn',
  },
  ar: {
    title: 'باراكودا - العب ضد المحرك',
    tag: 'محرك شطرنج بلغة C++20 · شبكة NNUE مدرّبة ذاتياً · يعمل داخل متصفحك',
    eloTitle: 'القوة التقديرية للشبكة المنشورة',
    evalTitle: 'التقييم (من جهة الأبيض)',
    boardLabel: 'رقعة الشطرنج',
    game: 'اللعبة', level: 'المستوى', youPlay: 'تلعب بـ', white: 'الأبيض', black: 'الأسود', random: 'عشوائي',
    newGame: 'لعبة جديدة', undo: 'تراجع', flip: 'اقلب الرقعة',
    engine: 'المحرك', thinking: 'التفكير', depth: 'العمق', speed: 'السرعة',
    network: 'الشبكة', evaluation: 'التقييم', trained: 'التدريب', netStrength: 'القوة التقديرية', updated: 'آخر تحديث',
    moves: 'النقلات',
    footer: 'يعمل بالكامل داخل متصفحك عبر WebAssembly — بلا خادم ولا حساب. شيفرة المحرك على GitHub',
    idle: 'خامل', thinkingNow: 'يفكّر…',
    loading: 'جارٍ تحميل المحرك…', engineThinking: 'باراكودا يفكّر…',
    yourMove: 'دورك', inCheck: ' — ملكك في كش',
    youWin: ' — فزت!', engineWins: ' — فاز باراكودا',
    engineError: 'خطأ في المحرك: ', startError: 'تعذّر تشغيل المحرك: ',
    netNnue: 'NNUE (شبكة عصبية)', netClassic: 'كلاسيكي (مبرمج يدوياً)', notYet: 'لم يتدرب بعد',
    trainedFmt: '{epochs} دورة تدريب، {m} مليون عينة', eloFmt: '\u2066~{elo} Elo\u2069', // LRI..PDI keeps the number run left-to-right
    promote: 'الترقية إلى',
    lvl1: 'مبتدئ', lvl2: 'عادي', lvl3: 'لاعب نادي', lvl4: 'متقدم', lvl5: 'خبير', lvl6: 'الحد الأقصى',
    'reason.checkmate': 'كش مات', 'reason.stalemate': 'جمود (بات)', 'reason.50-move rule': 'قاعدة الخمسين نقلة',
    'reason.insufficient material': 'مواد غير كافية', 'reason.threefold repetition': 'تكرار ثلاثي',
    langButton: 'English',
    'viz.title': 'الشبكة العصبية',
    'viz.thinking': 'يفكّر: الشبكة تقيّم مواضع من الخط الذي يحلّله المحرك',
    'viz.idle': 'عرض الشبكة للموضع الحالي',
    'viz.used': 'هذه الشبكة هي التي تلعب باسم باراكودا.',
    'viz.notUsed': 'للعرض فقط: لا يزال باراكودا يلعب بتقييمه الكلاسيكي حتى تتفوق هذه الشبكة عليه في الاختبار.',
    'viz.none': 'لم تُنشر شبكة بعد، فهذا رسم توضيحي لبنية الشبكة فقط.',
    'viz.l0': 'المدخلات · القطع', 'viz.l1': 'محوّل الخصائص', 'viz.l2': 'مخفية 1', 'viz.l3': 'مخفية 2', 'viz.l4': 'المخرج',
    'viz.sub0': '{n} فعّالة من 768', 'viz.sub1': 'عرض 32 من 512', 'viz.sub2': '32 عصبوناً', 'viz.sub4': 'التقييم (الأبيض)',
    netClassicViz: 'كلاسيكي · الشبكة قيد التدريب',
    'viz.s0': 'المدخلات', 'viz.s1': 'الخصائص', 'viz.s2': 'مخفية 1', 'viz.s3': 'مخفية 2', 'viz.s4': 'المخرج',
    'tab.game': 'اللعبة', 'tab.net': 'الشبكة', 'tab.moves': 'النقلات',
    'thanks.title': 'شكرًا لك!',
    'thanks.msg': 'لقد تم تعليم باراكودا على طريقة تفكيرك، وأصبح عقلك ضمن شبكته العصبية.',
    'thanks.plain': 'شكرًا لك على اللعب!',
    'thanks.sent': 'حُفظت مباراتك دون أي بيانات شخصية (النقلات والنتيجة فقط) وستُستخدم في تدريب باراكودا.',
    'thanks.off': 'المشاركة متوقفة، لذلك لم تُحفظ هذه المباراة.',
    'thanks.show': 'أظهر الرقعة',
    'share.label': 'شارك مبارياتي دون ذكر هويتي ليتعلّم منها باراكودا',
  },
};

let lang = 'en';
export const getLang = () => lang;
export const isRtl = () => lang === 'ar';

export function t(key, params) {
  let s = (STRINGS[lang] && STRINGS[lang][key]) ?? STRINGS.en[key] ?? key;
  if (params) for (const [k, v] of Object.entries(params)) s = s.replaceAll(`{${k}}`, String(v));
  return s;
}

// Static text: elements carry data-i18n (text), data-i18n-title or data-i18n-label (attributes).
export function applyStatic() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.title = t('title');
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-title]')) el.title = t(el.dataset.i18nTitle);
  for (const el of document.querySelectorAll('[data-i18n-label]')) el.setAttribute('aria-label', t(el.dataset.i18nLabel));
}

export function initLang() {
  let saved = null;
  try { saved = localStorage.getItem('barracuda-lang'); } catch (_) { /* storage unavailable */ }
  const guess = (navigator.language || 'en').toLowerCase().startsWith('ar') ? 'ar' : 'en';
  lang = saved === 'ar' || saved === 'en' ? saved : guess;
  applyStatic();
}

export function setLang(next) {
  lang = next === 'ar' ? 'ar' : 'en';
  try { localStorage.setItem('barracuda-lang', lang); } catch (_) { /* ignore */ }
  applyStatic();
}
